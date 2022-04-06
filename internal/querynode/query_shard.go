// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package querynode

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/golang/protobuf/proto"
	"go.uber.org/zap"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/distance"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

type queryShard struct {
	ctx    context.Context
	cancel context.CancelFunc

	collectionID UniqueID
	channel      Channel
	replicaID    int64

	clusterService *ShardClusterService
	historical     *historical
	streaming      *streaming

	localChunkManager  storage.ChunkManager
	remoteChunkManager storage.ChunkManager
	vectorChunkManager *storage.VectorChunkManager
	localCacheEnabled  bool
	localCacheSize     int64
}

func newQueryShard(
	ctx context.Context,
	collectionID UniqueID,
	channel Channel,
	replicaID int64,
	clusterService *ShardClusterService,
	historical *historical,
	streaming *streaming,
	localChunkManager storage.ChunkManager,
	remoteChunkManager storage.ChunkManager,
	localCacheEnabled bool,
) *queryShard {
	ctx, cancel := context.WithCancel(ctx)
	qs := &queryShard{
		ctx:                ctx,
		cancel:             cancel,
		collectionID:       collectionID,
		channel:            channel,
		replicaID:          replicaID,
		clusterService:     clusterService,
		historical:         historical,
		streaming:          streaming,
		localChunkManager:  localChunkManager,
		remoteChunkManager: remoteChunkManager,
		localCacheEnabled:  localCacheEnabled,
		localCacheSize:     Params.QueryNodeCfg.LocalFileCacheLimit,
	}
	return qs
}

func (q *queryShard) search(ctx context.Context, req *querypb.SearchRequest) (*internalpb.SearchResults, error) {
	collectionID := req.Req.CollectionID
	segmentIDs := req.SegmentIDs
	timestamp := req.Req.TravelTimestamp
	// TODO: hold request until guarantee timestamp >= service timestamp

	q.historical.replica.queryRLock()
	q.streaming.replica.queryRLock()
	defer q.historical.replica.queryRUnlock()
	defer q.streaming.replica.queryRLock()

	// deserialize query plan
	collection, err := q.historical.replica.getCollectionByID(collectionID)
	if err != nil {
		return nil, err
	}

	var plan *SearchPlan
	if req.Req.GetDslType() == commonpb.DslType_BoolExprV1 {
		expr := req.Req.SerializedExprPlan
		plan, err = createSearchPlanByExpr(collection, expr)
		if err != nil {
			return nil, err
		}
	} else {
		dsl := req.Req.Dsl
		plan, err = createSearchPlan(collection, dsl)
		if err != nil {
			return nil, err
		}
	}
	defer plan.delete()

	schemaHelper, err := typeutil.CreateSchemaHelper(collection.schema)
	if err != nil {
		return nil, err
	}

	// validate top-k
	topK := plan.getTopK()
	if topK <= 0 || topK >= 16385 {
		return nil, fmt.Errorf("limit should be in range [1, 16385], but got %d", topK)
	}

	// parse plan to search request
	searchReq, err := parseSearchRequest(plan, req.Req.PlaceholderGroup)
	if err != nil {
		return nil, err
	}
	defer searchReq.delete()
	queryNum := searchReq.getNumOfQuery()
	searchRequests := []*searchRequest{searchReq}

	if len(segmentIDs) == 0 {
		cluster, ok := q.clusterService.getShardCluster(req.GetDmlChannel())
		if !ok {
			return nil, fmt.Errorf("channel %s leader is not here", req.GetDmlChannel())
		}

		// shard leader dispatches request to its shard cluster
		results, err := cluster.Search(ctx, req)
		if err != nil {
			return nil, err
		}
		// shard leader queries its own streaming data
		streamingResults, _, _, err := q.streaming.search(searchRequests, collectionID, req.Req.PartitionIDs, req.DmlChannel, plan, timestamp)
		if err != nil {
			return nil, err
		}
		defer deleteSearchResults(streamingResults)

		// reduce search results
		numSegment := int64(len(streamingResults))
		err = reduceSearchResultsAndFillData(plan, streamingResults, numSegment)
		marshaledHits, err := reorganizeSearchResults(streamingResults, numSegment)
		if err != nil {
			return nil, err
		}
		defer deleteMarshaledHits(marshaledHits)

		// transform (hard to understand)
		hitsBlob, err := marshaledHits.getHitsBlob()
		if err != nil {
			return nil, err
		}

		hitBlobSizePeerQuery, err := marshaledHits.hitBlobSizeInGroup(int64(0))
		if err != nil {
			return nil, err
		}

		var offset int64 = 0
		hits := make([][]byte, len(hitBlobSizePeerQuery))
		for i, length := range hitBlobSizePeerQuery {
			hits[i] = hitsBlob[offset : offset+length]
			offset += length
		}

		transformed, err := translateHits(schemaHelper, req.Req.OutputFieldsId, hits)
		if err != nil {
			return nil, err
		}
		byteBlobs, err := proto.Marshal(transformed)
		if err != nil {
			return nil, err
		}

		// complete results with merged streaming result
		results = append(results, &internalpb.SearchResults{
			Status:         &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			MetricType:     plan.getMetricType(),
			NumQueries:     queryNum,
			TopK:           topK,
			SlicedBlob:     byteBlobs,
			SlicedOffset:   1,
			SlicedNumCount: 1,
		})

		// reduce shard search results: unmarshal -> reduce -> marshal
		searchResultData, err := decodeSearchResults(results)
		if err != nil {
			return nil, err
		}
		reducedResultData, err := reduceSearchResultData(searchResultData, queryNum, plan.getTopK(), plan.getMetricType())
		if err != nil {
			return nil, err
		}
		return encodeSearchResultData(reducedResultData)
	}

	// search each segments by segment IDs in request
	searchResults, _, err := q.historical.searchSegments(segmentIDs, searchRequests, plan, timestamp)
	if err != nil {
		return nil, err
	}
	defer deleteSearchResults(searchResults)

	// reduce search results
	numSegment := int64(len(searchResults))
	err = reduceSearchResultsAndFillData(plan, searchResults, numSegment)
	marshaledHits, err := reorganizeSearchResults(searchResults, numSegment)
	if err != nil {
		return nil, err
	}
	defer deleteMarshaledHits(marshaledHits)

	// transform (hard to understand)
	hitsBlob, err := marshaledHits.getHitsBlob()
	if err != nil {
		return nil, err
	}

	hitBlobSizePeerQuery, err := marshaledHits.hitBlobSizeInGroup(int64(0))
	if err != nil {
		return nil, err
	}

	var offset int64 = 0
	hits := make([][]byte, len(hitBlobSizePeerQuery))
	for i, length := range hitBlobSizePeerQuery {
		hits[i] = hitsBlob[offset : offset+length]
		offset += length
	}

	transformed, err := translateHits(schemaHelper, req.Req.OutputFieldsId, hits)
	if err != nil {
		return nil, err
	}
	byteBlobs, err := proto.Marshal(transformed)
	if err != nil {
		return nil, err
	}

	resp := &internalpb.SearchResults{
		Status:         &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		MetricType:     plan.getMetricType(),
		NumQueries:     queryNum,
		TopK:           topK,
		SlicedBlob:     byteBlobs,
		SlicedOffset:   1,
		SlicedNumCount: 1,
	}
	return resp, nil
}

func reduceSearchResultData(searchResultData []*schemapb.SearchResultData, nq int64, topk int64, metricType string) (*schemapb.SearchResultData, error) {
	ret := &schemapb.SearchResultData{
		NumQueries: nq,
		TopK:       topk,
		FieldsData: make([]*schemapb.FieldData, len(searchResultData[0].FieldsData)),
		Scores:     make([]float32, 0),
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: make([]int64, 0),
				},
			},
		},
		Topks: make([]int64, 0),
	}

	var skipDupCnt int64
	var realTopK int64 = -1
	for i := int64(0); i < nq; i++ {
		offsets := make([]int64, len(searchResultData))

		var idSet = make(map[int64]struct{})
		var j int64
		for j = 0; j < topk; {
			sel := selectSearchResultData(searchResultData, offsets, topk, i)
			if sel == -1 {
				break
			}
			idx := i*topk + offsets[sel]

			id := searchResultData[sel].Ids.GetIntId().Data[idx]
			score := searchResultData[sel].Scores[idx]
			// ignore invalid search result
			if id == -1 {
				continue
			}

			// remove duplicates
			if _, ok := idSet[id]; !ok {
				typeutil.AppendFieldData(ret.FieldsData, searchResultData[sel].FieldsData, idx)
				ret.Ids.GetIntId().Data = append(ret.Ids.GetIntId().Data, id)
				ret.Scores = append(ret.Scores, score)
				idSet[id] = struct{}{}
				j++
			} else {
				// skip entity with same id
				skipDupCnt++
			}
			offsets[sel]++
		}
		if realTopK != -1 && realTopK != j {
			log.Warn("Proxy Reduce Search Result", zap.Error(errors.New("the length (topk) between all result of query is different")))
			// return nil, errors.New("the length (topk) between all result of query is different")
		}
		realTopK = j
		ret.Topks = append(ret.Topks, realTopK)
	}
	log.Debug("skip duplicated search result", zap.Int64("count", skipDupCnt))
	ret.TopK = realTopK

	if !distance.PositivelyRelated(metricType) {
		for k := range ret.Scores {
			ret.Scores[k] *= -1
		}
	}

	return ret, nil
}

func selectSearchResultData(dataArray []*schemapb.SearchResultData, offsets []int64, topk int64, qi int64) int {
	sel := -1
	maxDistance := -1 * float32(math.MaxFloat32)
	for i, offset := range offsets { // query num, the number of ways to merge
		if offset >= topk {
			continue
		}
		idx := qi*topk + offset
		id := dataArray[i].Ids.GetIntId().Data[idx]
		if id != -1 {
			distance := dataArray[i].Scores[idx]
			if distance > maxDistance {
				sel = i
				maxDistance = distance
			}
		}
	}
	return sel
}

func decodeSearchResults(searchResults []*internalpb.SearchResults) ([]*schemapb.SearchResultData, error) {
	results := make([]*schemapb.SearchResultData, 0)
	for _, partialSearchResult := range searchResults {
		if partialSearchResult.SlicedBlob == nil {
			continue
		}

		var partialResultData schemapb.SearchResultData
		err := proto.Unmarshal(partialSearchResult.SlicedBlob, &partialResultData)
		if err != nil {
			return nil, err
		}

		results = append(results, &partialResultData)
	}
	return results, nil
}

func encodeSearchResultData(searchResultData *schemapb.SearchResultData) (searchResults *internalpb.SearchResults, err error) {
	searchResults.SlicedBlob, err = proto.Marshal(searchResultData)
	return
}

func (q *queryShard) query(ctx context.Context, req *querypb.QueryRequest) (*internalpb.RetrieveResults, error) {
	collectionID := req.Req.CollectionID
	segmentIDs := req.SegmentIDs
	partitionIDs := req.Req.PartitionIDs
	expr := req.Req.SerializedExprPlan
	timestamp := req.Req.TravelTimestamp
	// TODO: hold request until guarantee timestamp >= service timestamp

	// deserialize query plan
	collection, err := q.streaming.replica.getCollectionByID(collectionID)
	if err != nil {
		return nil, err
	}
	plan, err := createRetrievePlanByExpr(collection, expr, timestamp)
	if err != nil {
		return nil, err
	}
	defer plan.delete()

	// TODO: init vector chunk manager at most once
	if q.vectorChunkManager == nil {
		if q.localChunkManager == nil {
			return nil, fmt.Errorf("can not create vector chunk manager for local chunk manager is nil")
		}
		if q.remoteChunkManager == nil {
			return nil, fmt.Errorf("can not create vector chunk manager for remote chunk manager is nil")
		}
		q.vectorChunkManager, err = storage.NewVectorChunkManager(q.localChunkManager, q.remoteChunkManager,
			&etcdpb.CollectionMeta{
				ID:     collection.id,
				Schema: collection.schema,
			}, q.localCacheSize, q.localCacheEnabled)
		if err != nil {
			return nil, err
		}
	}

	// check if shard leader b.c only leader receives request with no segment specified
	if len(req.GetSegmentIDs()) == 0 {
		cluster, ok := q.clusterService.getShardCluster(req.GetDmlChannel())
		if !ok {
			return nil, fmt.Errorf("channel %s leader is not here", req.GetDmlChannel())
		}

		// shard leader dispatches request to its shard cluster
		results, err := cluster.Query(ctx, req)
		if err != nil {
			return nil, err
		}
		// shard leader queries its own streaming data
		// TODO: filter stream retrieve results by channel
		streamingResults, _, _, err := q.streaming.retrieve(collectionID, partitionIDs, plan)
		if err != nil {
			return nil, err
		}
		streamingResult, err := mergeRetrieveResults(streamingResults)
		if err != nil {
			return nil, err
		}
		// complete results with merged streaming result
		results = append(results, &internalpb.RetrieveResults{
			Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
			Ids:        streamingResult.Ids,
			FieldsData: streamingResult.FieldsData,
		})
		// merge shard query results
		return mergeInternalRetrieveResults(results)
	}

	// shard follower considers solely historical segments
	retrieveResults, err := q.historical.retrieveBySegmentIDs(collectionID, segmentIDs, q.vectorChunkManager, plan)
	if err != nil {
		return nil, err
	}
	mergedResult, err := mergeRetrieveResults(retrieveResults)
	if err != nil {
		return nil, err
	}

	log.Debug("retrieve result", zap.String("ids", mergedResult.Ids.String()))
	RetrieveResults := &internalpb.RetrieveResults{
		Status:     &commonpb.Status{ErrorCode: commonpb.ErrorCode_Success},
		Ids:        mergedResult.Ids,
		FieldsData: mergedResult.FieldsData,
	}
	return RetrieveResults, nil
}

// TODO: largely based on function mergeRetrieveResults, need rewriting
func mergeInternalRetrieveResults(retrieveResults []*internalpb.RetrieveResults) (*internalpb.RetrieveResults, error) {
	var ret *internalpb.RetrieveResults
	var skipDupCnt int64
	var idSet = make(map[int64]struct{})

	// merge results and remove duplicates
	for _, rr := range retrieveResults {

		if ret == nil {
			ret = &internalpb.RetrieveResults{
				Ids: &schemapb.IDs{
					IdField: &schemapb.IDs_IntId{
						IntId: &schemapb.LongArray{
							Data: []int64{},
						},
					},
				},
				FieldsData: make([]*schemapb.FieldData, len(rr.FieldsData)),
			}
		}

		if len(ret.FieldsData) != len(rr.FieldsData) {
			return nil, fmt.Errorf("mismatch FieldData in RetrieveResults")
		}

		dstIds := ret.Ids.GetIntId()
		for i, id := range rr.Ids.GetIntId().GetData() {
			if _, ok := idSet[id]; !ok {
				dstIds.Data = append(dstIds.Data, id)
				typeutil.AppendFieldData(ret.FieldsData, rr.FieldsData, int64(i))
				idSet[id] = struct{}{}
			} else {
				// primary keys duplicate
				skipDupCnt++
			}
		}
	}

	// not found, return default values indicating not result found
	if ret == nil {
		ret = &internalpb.RetrieveResults{
			Ids:        &schemapb.IDs{},
			FieldsData: []*schemapb.FieldData{},
		}
	}

	return ret, nil
}
