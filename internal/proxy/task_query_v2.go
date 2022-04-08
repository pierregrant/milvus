package proxy

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	qnClient "github.com/milvus-io/milvus/internal/distributed/querynode/client"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/internal/util/timerecord"
	"github.com/milvus-io/milvus/internal/util/tsoutil"
	"github.com/milvus-io/milvus/internal/util/typeutil"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"
)

type queryTaskV2 struct {
	Condition
	*internalpb.RetrieveRequest

	ctx            context.Context
	result         *milvuspb.QueryResults
	request        *milvuspb.QueryRequest
	qc             types.QueryCoord
	ids            *schemapb.IDs
	collectionName string

	resultBuf          chan *internalpb.RetrieveResults
	toReduceResults    []*internalpb.RetrieveResults
	runningGroup       *errgroup.Group
	runningGroupCtx    context.Context
	getQueryNodePolicy func(context.Context, string) (types.QueryNode, error)
}

func (t *queryTaskV2) PreExecute(ctx context.Context) error {
	t.Base.MsgType = commonpb.MsgType_Retrieve
	t.Base.SourceID = Params.ProxyCfg.ProxyID

	collectionName := t.request.CollectionName
	t.collectionName = collectionName
	if err := validateCollectionName(collectionName); err != nil {
		log.Warn("Invalid collection name.", zap.String("collectionName", collectionName),
			zap.Int64("requestID", t.Base.MsgID), zap.String("requestType", "query"))
		return err
	}

	log.Info("Validate collection name.", zap.Any("collectionName", collectionName),
		zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))

	collID, err := globalMetaCache.GetCollectionID(ctx, collectionName)
	if err != nil {
		log.Debug("Failed to get collection id.", zap.Any("collectionName", collectionName),
			zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))
		return err
	}

	t.CollectionID = collID
	log.Info("Get collection ID by name",
		zap.Int64("collectionID", t.CollectionID), zap.String("collection name", collectionName),
		zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))

	for _, tag := range t.request.PartitionNames {
		if err := validatePartitionTag(tag, false); err != nil {
			log.Warn("invalid partition name", zap.String("partition name", tag),
				zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))
			return err
		}
	}
	log.Debug("Validate partition names.",
		zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))

	t.PartitionIDs = make([]UniqueID, 0)
	partitionsMap, err := globalMetaCache.GetPartitions(ctx, collectionName)
	if err != nil {
		log.Warn("failed to get partitions in collection.", zap.String("collection name", collectionName),
			zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))
		return err
	}
	log.Debug("Get partitions in collection.", zap.Any("collectionName", collectionName),
		zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))

	// Check if partitions are valid partitions in collection
	partitionsRecord := make(map[UniqueID]bool)
	for _, partitionName := range t.request.PartitionNames {
		pattern := fmt.Sprintf("^%s$", partitionName)
		re, err := regexp.Compile(pattern)
		if err != nil {
			log.Debug("failed to compile partition name regex expression.", zap.Any("partition name", partitionName),
				zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))
			return errors.New("invalid partition names")
		}
		found := false
		for name, pID := range partitionsMap {
			if re.MatchString(name) {
				if _, exist := partitionsRecord[pID]; !exist {
					t.PartitionIDs = append(t.PartitionIDs, pID)
					partitionsRecord[pID] = true
				}
				found = true
			}
		}
		if !found {
			// FIXME(wxyu): undefined behavior
			errMsg := fmt.Sprintf("partition name: %s not found", partitionName)
			return errors.New(errMsg)
		}
	}

	if !t.checkIfLoaded(collID, t.PartitionIDs) {
		return fmt.Errorf("collection:%v or partition:%v not loaded into memory", collectionName, t.request.GetPartitionNames())
	}

	schema, _ := globalMetaCache.GetCollectionSchema(ctx, collectionName)

	if t.ids != nil {
		pkField := ""
		for _, field := range schema.Fields {
			if field.IsPrimaryKey {
				pkField = field.Name
			}
		}
		t.request.Expr = IDs2Expr(pkField, t.ids.GetIntId().Data)
	}

	if t.request.Expr == "" {
		return fmt.Errorf("query expression is empty")
	}

	plan, err := createExprPlan(schema, t.request.Expr)
	if err != nil {
		return err
	}
	t.request.OutputFields, err = translateOutputFields(t.request.OutputFields, schema, true)
	if err != nil {
		return err
	}
	log.Debug("translate output fields", zap.Any("OutputFields", t.request.OutputFields),
		zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))

	if len(t.request.OutputFields) == 0 {
		for _, field := range schema.Fields {
			if field.FieldID >= 100 && field.DataType != schemapb.DataType_FloatVector && field.DataType != schemapb.DataType_BinaryVector {
				t.OutputFieldsId = append(t.OutputFieldsId, field.FieldID)
			}
		}
	} else {
		addPrimaryKey := false
		for _, reqField := range t.request.OutputFields {
			findField := false
			for _, field := range schema.Fields {
				if reqField == field.Name {
					if field.IsPrimaryKey {
						addPrimaryKey = true
					}
					findField = true
					t.OutputFieldsId = append(t.OutputFieldsId, field.FieldID)
					plan.OutputFieldIds = append(plan.OutputFieldIds, field.FieldID)
				} else {
					if field.IsPrimaryKey && !addPrimaryKey {
						t.OutputFieldsId = append(t.OutputFieldsId, field.FieldID)
						plan.OutputFieldIds = append(plan.OutputFieldIds, field.FieldID)
						addPrimaryKey = true
					}
				}
			}
			if !findField {
				return fmt.Errorf("field %s not exist", reqField)
			}
		}
	}
	log.Debug("translate output fields to field ids", zap.Any("OutputFieldsID", t.OutputFieldsId),
		zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))

	t.RetrieveRequest.SerializedExprPlan, err = proto.Marshal(plan)
	if err != nil {
		return err
	}

	if t.request.TravelTimestamp == 0 {
		t.TravelTimestamp = t.BeginTs()
	} else {
		durationSeconds := tsoutil.CalculateDuration(t.BeginTs(), t.request.TravelTimestamp) / 1000
		if durationSeconds > Params.CommonCfg.RetentionDuration {
			duration := time.Second * time.Duration(durationSeconds)
			return fmt.Errorf("only support to travel back to %s so far", duration.String())
		}
		t.TravelTimestamp = t.request.TravelTimestamp
	}

	if t.request.GuaranteeTimestamp == 0 {
		t.GuaranteeTimestamp = t.BeginTs()
	} else {
		t.GuaranteeTimestamp = t.request.GuaranteeTimestamp
	}

	deadline, ok := t.TraceCtx().Deadline()
	if ok {
		t.TimeoutTimestamp = tsoutil.ComposeTSByTime(deadline, 0)
	}

	t.DbID = 0 // TODO
	if t.getQueryNodePolicy == nil {
		t.getQueryNodePolicy = defaultGetQueryNodePolicy
	}

	log.Info("Query PreExecute done.",
		zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))
	return nil
}

func (t *queryTaskV2) Execute(ctx context.Context) error {
	tr := timerecord.NewTimeRecorder(fmt.Sprintf("proxy execute query %d", t.ID()))
	defer tr.Elapse("done")

	req := &querypb.GetShardLeadersRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_GetShardLeaders,
			MsgID:     t.Base.MsgID,
			Timestamp: t.Base.Timestamp,
			SourceID:  Params.ProxyCfg.ProxyID,
		},
		CollectionID: t.CollectionID,
	}
	resp, err := t.qc.GetShardLeaders(ctx, req)
	if err != nil {
		return err
	}
	if resp.Status.ErrorCode != commonpb.ErrorCode_Success {
		return fmt.Errorf("fail to get shard leaders from QueryCoord: %s", resp.Status.Reason)
	}

	shards := resp.GetShards()
	t.resultBuf = make(chan *internalpb.RetrieveResults, len(shards))
	t.toReduceResults = make([]*internalpb.RetrieveResults, 0, len(shards))

	t.runningGroup, t.runningGroupCtx = errgroup.WithContext(ctx)
	for _, shard := range shards {
		s := shard
		t.runningGroup.Go(func() error {
			log.Debug("proxy starting to query one shard",
				zap.Int64("collectionID", t.CollectionID),
				zap.String("collection name", t.collectionName),
				zap.String("shard channel", s.GetChannelName()),
				zap.Uint64("timeoutTs", t.TimeoutTimestamp))

			err := t.queryShard(t.runningGroupCtx, s)
			if err != nil {
				return err
			}
			return nil
		})
	}

	log.Info("Query Execute done.",
		zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))
	return nil
}

func (t *queryTaskV2) PostExecute(ctx context.Context) error {
	tr := timerecord.NewTimeRecorder("queryTask PostExecute")
	defer func() {
		tr.Elapse("done")
	}()

	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		for {
			select {
			case <-t.TraceCtx().Done():
				log.Warn("proxy", zap.Int64("Query: wait to finish failed, timeout!, taskID:", t.ID()))
				return
			case <-t.runningGroupCtx.Done():
				log.Debug("all queries are finished or canceled", zap.Any("taskID", t.ID()))
				close(t.resultBuf)
				for res := range t.resultBuf {
					t.toReduceResults = append(t.toReduceResults, res)
					log.Debug("proxy receives one query result", zap.Int64("sourceID", res.GetBase().GetSourceID()), zap.Any("taskID", t.ID()))
				}
				wg.Done()
				return
			}
		}
	}()

	err := t.runningGroup.Wait()
	if err != nil {
		return err
	}

	wg.Wait()
	t.result, err = mergeRetrieveResults(t.toReduceResults)
	if err != nil {
		return err
	}
	t.result.CollectionName = t.collectionName

	if len(t.result.FieldsData) > 0 {
		t.result.Status = &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_Success,
		}
	} else {
		log.Info("Query result is nil", zap.Any("requestID", t.Base.MsgID), zap.Any("requestType", "query"))
		t.result.Status = &commonpb.Status{
			ErrorCode: commonpb.ErrorCode_EmptyCollection,
			Reason:    "emptly collection", // TODO
		}
		return nil
	}

	schema, err := globalMetaCache.GetCollectionSchema(ctx, t.request.CollectionName)
	if err != nil {
		return err
	}
	for i := 0; i < len(t.result.FieldsData); i++ {
		for _, field := range schema.Fields {
			if field.FieldID == t.OutputFieldsId[i] {
				t.result.FieldsData[i].FieldName = field.Name
				t.result.FieldsData[i].FieldId = field.FieldID
				t.result.FieldsData[i].Type = field.DataType
			}
		}
	}
	log.Info("Query PostExecute done", zap.Any("requestID", t.Base.MsgID), zap.String("requestType", "query"))
	return nil
}

func (t *queryTaskV2) queryShard(ctx context.Context, leaders *querypb.ShardLeadersList) error {
	query := func(nodeID UniqueID, qn types.QueryNode) error {
		req := &querypb.QueryRequest{
			Req:        t.RetrieveRequest,
			DmlChannel: leaders.GetChannelName(),
		}

		result, err := qn.Query(ctx, req)
		if err != nil {
			log.Warn("QueryNode query returns error", zap.Int64("nodeID", nodeID),
				zap.Error(err))
			return fmt.Errorf("fail to Query, QueryNodeID=%d, err=%s", nodeID, err.Error())
		}
		if result.GetStatus().GetErrorCode() != commonpb.ErrorCode_Success {
			log.Warn("QueryNode query result error", zap.Int64("nodeID", nodeID),
				zap.String("reason", result.GetStatus().GetReason()))
			return fmt.Errorf("fail to Query, QueryNode ID = %d, reason=%s", nodeID, result.GetStatus().GetReason())
		}

		log.Debug("get query result", zap.Int64("nodeID", nodeID), zap.String("channelID", leaders.GetChannelName()))
		t.resultBuf <- result
		return nil
	}

	err := t.RoundRobin(query, leaders)
	if err != nil {
		log.Warn("fail to Query to all shard leaders", zap.Any("shard leaders", leaders.GetNodeIds()))
		return err
	}

	return nil
}

// TODO add another policy to enbale the use of cache
// defaultGetQueryNodePolicy creates QueryNode client for every address everytime
func defaultGetQueryNodePolicy(ctx context.Context, address string) (types.QueryNode, error) {
	qn, err := qnClient.NewClient(ctx, address)
	if err != nil {
		return nil, err
	}

	if err := qn.Init(); err != nil {
		return nil, err
	}

	if err := qn.Start(); err != nil {
		return nil, err
	}
	return qn, nil
}

var errBegin = errors.New("begin error")

func (t *queryTaskV2) RoundRobin(query func(UniqueID, types.QueryNode) error, leaders *querypb.ShardLeadersList) error {
	var (
		err     = errBegin
		current = 0
		qn      types.QueryNode
	)
	replicaNum := len(leaders.GetNodeIds())

	for err != nil && current < replicaNum {
		currentID := leaders.GetNodeIds()[current]
		if err != errBegin {
			log.Warn("retry with another QueryNode", zap.String("leader", leaders.GetChannelName()), zap.Int64("nodeID", currentID))
		}

		qn, err = t.getQueryNodePolicy(t.TraceCtx(), leaders.GetNodeAddrs()[current])
		if err != nil {
			log.Warn("fail to get valid QueryNode", zap.Int64("nodeID", currentID),
				zap.Error(err))
			current++
			continue
		}

		defer qn.Stop()
		err = query(currentID, qn)
		if err != nil {
			log.Warn("fail to Query with shard leader",
				zap.String("leader", leaders.GetChannelName()),
				zap.Int64("nodeID", currentID),
				zap.Error(err))
		}
		current++
	}

	if current == replicaNum && err != nil {
		return fmt.Errorf("no shard leaders available for channel: %s, leaders: %v", leaders.GetChannelName(), leaders.GetNodeIds())
	}
	return nil
}

func (t *queryTaskV2) checkIfLoaded(collectionID UniqueID, partitionIDs []UniqueID) bool {
	resp, err := t.qc.ShowCollections(t.ctx, &querypb.ShowCollectionsRequest{
		Base: &commonpb.MsgBase{
			MsgType:   commonpb.MsgType_ShowCollections,
			MsgID:     t.Base.MsgID,
			Timestamp: t.Base.Timestamp,
			SourceID:  Params.ProxyCfg.ProxyID,
		},
	})
	if err != nil {
		log.Warn("fail to show collections by QueryCoord",
			zap.Int64("requestID", t.Base.MsgID), zap.String("requestType", "query"),
			zap.Error(err))
		return false
	}

	if resp.Status.ErrorCode != commonpb.ErrorCode_Success {
		log.Warn("fail to show collections by QueryCoord",
			zap.Int64("requestID", t.Base.MsgID), zap.String("requestType", "query"),
			zap.String("reason", resp.GetStatus().GetReason()))
		return false
	}

	loaded := true
	for index, collID := range resp.CollectionIDs {
		if collID == collectionID && resp.GetInMemoryPercentages()[index] >= int64(100) {
			loaded = false
			break
		}
	}

	if !loaded && len(partitionIDs) > 0 {
		resp, err := t.qc.ShowPartitions(t.ctx, &querypb.ShowPartitionsRequest{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_ShowCollections,
				MsgID:     t.Base.MsgID,
				Timestamp: t.Base.Timestamp,
				SourceID:  Params.ProxyCfg.ProxyID,
			},
			CollectionID: collectionID,
			PartitionIDs: partitionIDs,
		})
		if err != nil {
			log.Warn("fail to show partitions by QueryCoord",
				zap.Int64("requestID", t.Base.MsgID),
				zap.Int64("collectionID", collectionID),
				zap.Int64s("partitionIDs", partitionIDs),
				zap.String("requestType", "search"),
				zap.Error(err))
			return false
		}

		if resp.Status.ErrorCode != commonpb.ErrorCode_Success {
			log.Warn("fail to show partitions by QueryCoord",
				zap.Int64("collectionID", collectionID),
				zap.Int64s("partitionIDs", partitionIDs),
				zap.Int64("requestID", t.Base.MsgID), zap.String("requestType", "search"),
				zap.String("reason", resp.GetStatus().GetReason()))
			return false
		}
		// Current logic: show partitions won't return error if the given partitions are all loaded
		return true

	}
	return loaded
}

// IDs2Expr converts ids slices to bool expresion with specified field name
func IDs2Expr(fieldName string, ids []int64) string {
	idsStr := strings.Trim(strings.Join(strings.Fields(fmt.Sprint(ids)), ", "), "[]")
	return fieldName + " in [ " + idsStr + " ]"
}

func mergeRetrieveResults(retrieveResults []*internalpb.RetrieveResults) (*milvuspb.QueryResults, error) {
	var ret *milvuspb.QueryResults
	var skipDupCnt int64
	var idSet = make(map[int64]struct{})

	// merge results and remove duplicates
	for _, rr := range retrieveResults {
		// skip empty result, it will break merge result
		if rr == nil || rr.Ids == nil || rr.Ids.GetIntId() == nil || len(rr.Ids.GetIntId().Data) == 0 {
			continue
		}

		if ret == nil {
			ret = &milvuspb.QueryResults{
				FieldsData: make([]*schemapb.FieldData, len(rr.FieldsData)),
			}
		}

		if len(ret.FieldsData) != len(rr.FieldsData) {
			return nil, fmt.Errorf("mismatch FieldData in proxy RetrieveResults, expect %d get %d", len(ret.FieldsData), len(rr.FieldsData))
		}

		for i, id := range rr.Ids.GetIntId().GetData() {
			if _, ok := idSet[id]; !ok {
				typeutil.AppendFieldData(ret.FieldsData, rr.FieldsData, int64(i))
				idSet[id] = struct{}{}
			} else {
				// primary keys duplicate
				skipDupCnt++
			}
		}
	}
	log.Debug("skip duplicated query result", zap.Int64("count", skipDupCnt))

	if ret == nil {
		ret = &milvuspb.QueryResults{
			FieldsData: []*schemapb.FieldData{},
		}
	}

	return ret, nil
}

func (t *queryTaskV2) TraceCtx() context.Context {
	return t.ctx
}

func (t *queryTaskV2) ID() UniqueID {
	return t.Base.MsgID
}

func (t *queryTaskV2) SetID(uid UniqueID) {
	t.Base.MsgID = uid
}

func (t *queryTaskV2) Name() string {
	return RetrieveTaskName
}

func (t *queryTaskV2) Type() commonpb.MsgType {
	return t.Base.MsgType
}

func (t *queryTaskV2) BeginTs() Timestamp {
	return t.Base.Timestamp
}

func (t *queryTaskV2) EndTs() Timestamp {
	return t.Base.Timestamp
}

func (t *queryTaskV2) SetTs(ts Timestamp) {
	t.Base.Timestamp = ts
}

func (t *queryTaskV2) OnEnqueue() error {
	t.Base.MsgType = commonpb.MsgType_Retrieve
	return nil
}
