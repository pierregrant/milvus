package proxy

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/types"

	"github.com/milvus-io/milvus/internal/proto/commonpb"
	"github.com/milvus-io/milvus/internal/proto/internalpb"
	"github.com/milvus-io/milvus/internal/proto/milvuspb"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/proto/schemapb"

	"github.com/milvus-io/milvus/internal/util/distance"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/milvus-io/milvus/internal/util/timerecord"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

const (
	testShardsNum = int32(2)
)

func TestSearchTaskV2_PostExecute(t *testing.T) {
	t.Run("Test empty result", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		qt := &searchTaskV2{
			ctx:       ctx,
			Condition: NewTaskCondition(context.TODO()),
			SearchRequest: &internalpb.SearchRequest{
				Base: &commonpb.MsgBase{
					MsgType:  commonpb.MsgType_Search,
					SourceID: Params.ProxyCfg.ProxyID,
				},
			},
			request: nil,
			qc:      nil,
			tr:      timerecord.NewTimeRecorder("search"),

			resultBuf:       make(chan *internalpb.SearchResults, 10),
			toReduceResults: make([]*internalpb.SearchResults, 0),
		}
		// no result
		qt.resultBuf <- &internalpb.SearchResults{}

		qt.runningGroup, qt.runningGroupCtx = errgroup.WithContext(ctx)
		for i := 0; i <= 2; i++ {
			num := i
			qt.runningGroup.Go(func() error {
				log.Debug("returning error groups", zap.Int("No.", num))
				return nil
			})
		}

		err := qt.PostExecute(context.TODO())
		assert.NoError(t, err)
		assert.Equal(t, qt.result.Status.ErrorCode, commonpb.ErrorCode_Success)
	})
}

func createColl(t *testing.T, name string, rc types.RootCoord) {
	schema := constructCollectionSchema(testInt64Field, testFloatVecField, testVecDim, name)
	marshaledSchema, err := proto.Marshal(schema)
	require.NoError(t, err)
	ctx := context.TODO()

	createColT := &createCollectionTask{
		Condition: NewTaskCondition(context.TODO()),
		CreateCollectionRequest: &milvuspb.CreateCollectionRequest{
			CollectionName: name,
			Schema:         marshaledSchema,
			ShardsNum:      testShardsNum,
		},
		ctx:       ctx,
		rootCoord: rc,
	}

	require.NoError(t, createColT.OnEnqueue())
	require.NoError(t, createColT.PreExecute(ctx))
	require.NoError(t, createColT.Execute(ctx))
	require.NoError(t, createColT.PostExecute(ctx))
}

func getValidSearchParams() []*commonpb.KeyValuePair {
	return []*commonpb.KeyValuePair{
		{
			Key:   AnnsFieldKey,
			Value: testFloatVecField,
		},
		{
			Key:   TopKKey,
			Value: "10",
		},
		{
			Key:   MetricTypeKey,
			Value: distance.L2,
		},
		{
			Key:   SearchParamsKey,
			Value: `{"nprobe": 10}`,
		},
		{
			Key:   RoundDecimalKey,
			Value: "-1",
		}}
}

func TestSearchTaskV2_PreExecute(t *testing.T) {
	var err error

	Params.Init()
	var (
		rc  = NewRootCoordMock()
		qc  = NewQueryCoordMock()
		ctx = context.TODO()

		collectionName = t.Name() + funcutil.GenRandomStr()
	)

	err = rc.Start()
	defer rc.Stop()
	require.NoError(t, err)
	err = InitMetaCache(rc)
	require.NoError(t, err)

	err = qc.Start()
	defer qc.Stop()
	require.NoError(t, err)

	getSearchTask := func(t *testing.T, collName string) *searchTaskV2 {
		task := &searchTaskV2{
			ctx:           ctx,
			SearchRequest: &internalpb.SearchRequest{},
			request: &milvuspb.SearchRequest{
				CollectionName: collName,
			},
			qc: qc,
			tr: timerecord.NewTimeRecorder("test-search"),
		}
		require.NoError(t, task.OnEnqueue())
		return task
	}

	t.Run("collection not exist", func(t *testing.T) {
		task := getSearchTask(t, collectionName)
		err = task.PreExecute(ctx)
		assert.Error(t, err)
	})

	t.Run("invalid collection name", func(t *testing.T) {
		task := getSearchTask(t, collectionName)
		createColl(t, collectionName, rc)

		invalidCollNameTests := []struct {
			inCollName  string
			description string
		}{
			{"$", "invalid collection name $"},
			{"0", "invalid collection name 0"},
		}

		for _, test := range invalidCollNameTests {
			t.Run(test.description, func(t *testing.T) {
				task.request.CollectionName = test.inCollName
				assert.Error(t, task.PreExecute(context.TODO()))
			})
		}
	})

	t.Run("invalid partition names", func(t *testing.T) {
		task := getSearchTask(t, collectionName)
		createColl(t, collectionName, rc)

		invalidCollNameTests := []struct {
			inPartNames []string
			description string
		}{
			{[]string{"$"}, "invalid partition name $"},
			{[]string{"0"}, "invalid collection name 0"},
			{[]string{"default", "$"}, "invalid empty partition name"},
		}

		for _, test := range invalidCollNameTests {
			t.Run(test.description, func(t *testing.T) {
				task.request.PartitionNames = test.inPartNames
				assert.Error(t, task.PreExecute(context.TODO()))
			})
		}
	})

	t.Run("test checkIfLoaded error", func(t *testing.T) {
		collName := "test_checkIfLoaded_error" + funcutil.GenRandomStr()
		createColl(t, collName, rc)
		collID, err := globalMetaCache.GetCollectionID(context.TODO(), collName)
		require.NoError(t, err)
		task := getSearchTask(t, collName)

		t.Run("show collection err", func(t *testing.T) {
			qc.SetShowCollectionsFunc(func(ctx context.Context, request *querypb.ShowCollectionsRequest) (*querypb.ShowCollectionsResponse, error) {
				return nil, errors.New("mock")
			})

			assert.False(t, task.checkIfLoaded(collID, []UniqueID{}))
		})

		t.Run("show collection status unexpected error", func(t *testing.T) {
			qc.SetShowCollectionsFunc(func(ctx context.Context, request *querypb.ShowCollectionsRequest) (*querypb.ShowCollectionsResponse, error) {
				return &querypb.ShowCollectionsResponse{
					Status: &commonpb.Status{
						ErrorCode: commonpb.ErrorCode_UnexpectedError,
						Reason:    "mock",
					},
				}, nil
			})

			assert.False(t, task.checkIfLoaded(collID, []UniqueID{}))
			assert.Error(t, task.PreExecute(ctx))
			qc.ResetShowCollectionsFunc()
		})

		t.Run("show partition error", func(t *testing.T) {
			qc.SetShowPartitionsFunc(func(ctx context.Context, req *querypb.ShowPartitionsRequest) (*querypb.ShowPartitionsResponse, error) {
				return &querypb.ShowPartitionsResponse{
					Status: &commonpb.Status{
						ErrorCode: commonpb.ErrorCode_UnexpectedError,
						Reason:    "mock",
					},
				}, nil
			})
			assert.False(t, task.checkIfLoaded(collID, []UniqueID{1}))
		})

		t.Run("show partition status unexpected error", func(t *testing.T) {
			qc.SetShowPartitionsFunc(func(ctx context.Context, req *querypb.ShowPartitionsRequest) (*querypb.ShowPartitionsResponse, error) {
				return nil, errors.New("mock error")
			})
			assert.False(t, task.checkIfLoaded(collID, []UniqueID{1}))
		})

		t.Run("show partitions success", func(t *testing.T) {
			qc.SetShowPartitionsFunc(func(ctx context.Context, req *querypb.ShowPartitionsRequest) (*querypb.ShowPartitionsResponse, error) {
				return &querypb.ShowPartitionsResponse{
					Status: &commonpb.Status{
						ErrorCode: commonpb.ErrorCode_Success,
					},
				}, nil
			})
			assert.True(t, task.checkIfLoaded(collID, []UniqueID{1}))
			qc.ResetShowPartitionsFunc()
		})

		t.Run("show collection success but not loaded", func(t *testing.T) {
			qc.SetShowCollectionsFunc(func(ctx context.Context, request *querypb.ShowCollectionsRequest) (*querypb.ShowCollectionsResponse, error) {
				return &querypb.ShowCollectionsResponse{
					Status: &commonpb.Status{
						ErrorCode: commonpb.ErrorCode_Success,
					},
					CollectionIDs:       []UniqueID{collID},
					InMemoryPercentages: []int64{0},
				}, nil
			})

			qc.SetShowPartitionsFunc(func(ctx context.Context, req *querypb.ShowPartitionsRequest) (*querypb.ShowPartitionsResponse, error) {
				return nil, errors.New("mock error")
			})
			assert.False(t, task.checkIfLoaded(collID, []UniqueID{}))

			qc.SetShowPartitionsFunc(func(ctx context.Context, req *querypb.ShowPartitionsRequest) (*querypb.ShowPartitionsResponse, error) {
				return nil, errors.New("mock error")
			})
			assert.False(t, task.checkIfLoaded(collID, []UniqueID{}))

			qc.SetShowPartitionsFunc(func(ctx context.Context, req *querypb.ShowPartitionsRequest) (*querypb.ShowPartitionsResponse, error) {
				return &querypb.ShowPartitionsResponse{
					Status: &commonpb.Status{
						ErrorCode: commonpb.ErrorCode_Success,
					},
					PartitionIDs: []UniqueID{1},
				}, nil
			})
			assert.True(t, task.checkIfLoaded(collID, []UniqueID{}))
		})

		qc.ResetShowCollectionsFunc()
		qc.ResetShowPartitionsFunc()
	})

	t.Run("invalid key value pairs", func(t *testing.T) {
		spNoTopk := []*commonpb.KeyValuePair{{
			Key:   AnnsFieldKey,
			Value: testFloatVecField}}

		spInvalidTopk := append(spNoTopk, &commonpb.KeyValuePair{
			Key:   TopKKey,
			Value: "invalid",
		})

		spNoMetricType := append(spNoTopk, &commonpb.KeyValuePair{
			Key:   TopKKey,
			Value: "10",
		})

		spNoSearchParams := append(spNoMetricType, &commonpb.KeyValuePair{
			Key:   MetricTypeKey,
			Value: distance.L2,
		})

		spNoRoundDecimal := append(spNoSearchParams, &commonpb.KeyValuePair{
			Key:   SearchParamsKey,
			Value: `{"nprobe": 10}`,
		})

		spInvalidRoundDecimal := append(spNoRoundDecimal, &commonpb.KeyValuePair{
			Key:   RoundDecimalKey,
			Value: "invalid",
		})

		tests := []struct {
			description   string
			invalidParams []*commonpb.KeyValuePair
		}{
			{"No_topk", spNoTopk},
			{"Invalid_topk", spInvalidTopk},
			{"No_Metric_type", spNoMetricType},
			{"No_search_params", spNoSearchParams},
			{"no_round_decimal", spNoRoundDecimal},
			{"Invalid_round_decimal", spInvalidRoundDecimal},
		}

		for _, test := range tests {
			t.Run(test.description, func(t *testing.T) {
				collName := "collection_" + test.description
				createColl(t, collName, rc)
				collID, err := globalMetaCache.GetCollectionID(context.TODO(), collName)
				require.NoError(t, err)
				task := getSearchTask(t, collName)
				task.request.DslType = commonpb.DslType_BoolExprV1

				status, err := qc.LoadCollection(ctx, &querypb.LoadCollectionRequest{
					Base: &commonpb.MsgBase{
						MsgType: commonpb.MsgType_LoadCollection,
					},
					CollectionID: collID,
				})
				require.NoError(t, err)
				require.Equal(t, commonpb.ErrorCode_Success, status.GetErrorCode())
				assert.Error(t, task.PreExecute(ctx))
			})
		}
	})

	t.Run("search with timeout", func(t *testing.T) {
		collName := "search_with_timeout" + funcutil.GenRandomStr()
		createColl(t, collName, rc)
		collID, err := globalMetaCache.GetCollectionID(context.TODO(), collName)
		require.NoError(t, err)
		status, err := qc.LoadCollection(ctx, &querypb.LoadCollectionRequest{
			Base: &commonpb.MsgBase{
				MsgType: commonpb.MsgType_LoadCollection,
			},
			CollectionID: collID,
		})
		require.NoError(t, err)
		require.Equal(t, commonpb.ErrorCode_Success, status.GetErrorCode())

		task := getSearchTask(t, collName)
		task.request.SearchParams = getValidSearchParams()
		task.request.DslType = commonpb.DslType_BoolExprV1

		ctxTimeout, cancel := context.WithTimeout(ctx, time.Second)
		defer cancel()
		require.Equal(t, typeutil.ZeroTimestamp, task.TimeoutTimestamp)

		task.ctx = ctxTimeout
		assert.NoError(t, task.PreExecute(ctx))
		assert.Greater(t, task.TimeoutTimestamp, typeutil.ZeroTimestamp)

		// field not exist
		task.ctx = context.TODO()
		task.request.OutputFields = []string{testInt64Field + funcutil.GenRandomStr()}
		assert.Error(t, task.PreExecute(ctx))

		// contain vector field
		task.request.OutputFields = []string{testFloatVecField}
		assert.Error(t, task.PreExecute(ctx))
	})
}

func TestSearchTaskV2_Execute(t *testing.T) {
	Params.Init()

	var (
		err error

		rc  = NewRootCoordMock()
		qc  = NewQueryCoordMock()
		ctx = context.TODO()

		collectionName = t.Name() + funcutil.GenRandomStr()
	)

	err = rc.Start()
	require.NoError(t, err)
	defer rc.Stop()
	err = InitMetaCache(rc)
	require.NoError(t, err)

	err = qc.Start()
	require.NoError(t, err)
	defer qc.Stop()

	task := &searchTaskV2{
		ctx: ctx,
		SearchRequest: &internalpb.SearchRequest{
			Base: &commonpb.MsgBase{
				MsgType:   commonpb.MsgType_Search,
				Timestamp: uint64(time.Now().UnixNano()),
			},
		},
		request: &milvuspb.SearchRequest{
			CollectionName: collectionName,
		},
		result: &milvuspb.SearchResults{
			Status: &commonpb.Status{},
		},
		qc: qc,
		tr: timerecord.NewTimeRecorder("search"),
	}
	require.NoError(t, task.OnEnqueue())
	createColl(t, collectionName, rc)
}

func genSearchResultData(nq int64, topk int64, ids []int64, scores []float32) *schemapb.SearchResultData {
	return &schemapb.SearchResultData{
		NumQueries: nq,
		TopK:       topk,
		FieldsData: nil,
		Scores:     scores,
		Ids: &schemapb.IDs{
			IdField: &schemapb.IDs_IntId{
				IntId: &schemapb.LongArray{
					Data: ids,
				},
			},
		},
		Topks: make([]int64, nq),
	}
}

func TestSearchTaskV2_Ts(t *testing.T) {
	Params.Init()
	task := &searchTaskV2{
		SearchRequest: &internalpb.SearchRequest{},

		tr: timerecord.NewTimeRecorder("test-search"),
	}
	require.NoError(t, task.OnEnqueue())

	ts := Timestamp(time.Now().Nanosecond())
	task.SetTs(ts)
	assert.Equal(t, ts, task.BeginTs())
	assert.Equal(t, ts, task.EndTs())
}

func TestSearchTaskV2_Reduce(t *testing.T) {
	// const (
	//     nq         = 1
	//     topk       = 4
	//     metricType = "L2"
	// )
	// t.Run("case1", func(t *testing.T) {
	//     ids := []int64{1, 2, 3, 4}
	//     scores := []float32{-1.0, -2.0, -3.0, -4.0}
	//     data1 := genSearchResultData(nq, topk, ids, scores)
	//     data2 := genSearchResultData(nq, topk, ids, scores)
	//     dataArray := make([]*schemapb.SearchResultData, 0)
	//     dataArray = append(dataArray, data1)
	//     dataArray = append(dataArray, data2)
	//     res, err := reduceSearchResultData(dataArray, nq, topk, metricType)
	//     assert.Nil(t, err)
	//     assert.Equal(t, ids, res.Results.Ids.GetIntId().Data)
	//     assert.Equal(t, []float32{1.0, 2.0, 3.0, 4.0}, res.Results.Scores)
	// })
	// t.Run("case2", func(t *testing.T) {
	//     ids1 := []int64{1, 2, 3, 4}
	//     scores1 := []float32{-1.0, -2.0, -3.0, -4.0}
	//     ids2 := []int64{5, 1, 3, 4}
	//     scores2 := []float32{-1.0, -1.0, -3.0, -4.0}
	//     data1 := genSearchResultData(nq, topk, ids1, scores1)
	//     data2 := genSearchResultData(nq, topk, ids2, scores2)
	//     dataArray := make([]*schemapb.SearchResultData, 0)
	//     dataArray = append(dataArray, data1)
	//     dataArray = append(dataArray, data2)
	//     res, err := reduceSearchResultData(dataArray, nq, topk, metricType)
	//     assert.Nil(t, err)
	//     assert.ElementsMatch(t, []int64{1, 5, 2, 3}, res.Results.Ids.GetIntId().Data)
	// })
}

func TestSearchTaskV2WithInvalidRoundDecimal(t *testing.T) {
	// var err error
	//
	// Params.Init()
	// Params.ProxyCfg.SearchResultChannelNames = []string{funcutil.GenRandomStr()}
	//
	// rc := NewRootCoordMock()
	// rc.Start()
	// defer rc.Stop()
	//
	// ctx := context.Background()
	//
	// err = InitMetaCache(rc)
	// assert.NoError(t, err)
	//
	// shardsNum := int32(2)
	// prefix := "TestSearchTaskV2_all"
	// collectionName := prefix + funcutil.GenRandomStr()
	//
	// dim := 128
	// expr := fmt.Sprintf("%s > 0", testInt64Field)
	// nq := 10
	// topk := 10
	// roundDecimal := 7
	// nprobe := 10
	//
	// fieldName2Types := map[string]schemapb.DataType{
	//     testBoolField:     schemapb.DataType_Bool,
	//     testInt32Field:    schemapb.DataType_Int32,
	//     testInt64Field:    schemapb.DataType_Int64,
	//     testFloatField:    schemapb.DataType_Float,
	//     testDoubleField:   schemapb.DataType_Double,
	//     testFloatVecField: schemapb.DataType_FloatVector,
	// }
	// if enableMultipleVectorFields {
	//     fieldName2Types[testBinaryVecField] = schemapb.DataType_BinaryVector
	// }
	// schema := constructCollectionSchemaByDataType(collectionName, fieldName2Types, testInt64Field, false)
	// marshaledSchema, err := proto.Marshal(schema)
	// assert.NoError(t, err)
	//
	// createColT := &createCollectionTask{
	//     Condition: NewTaskCondition(ctx),
	//     CreateCollectionRequest: &milvuspb.CreateCollectionRequest{
	//         Base:           nil,
	//         CollectionName: collectionName,
	//         Schema:         marshaledSchema,
	//         ShardsNum:      shardsNum,
	//     },
	//     ctx:       ctx,
	//     rootCoord: rc,
	//     result:    nil,
	//     schema:    nil,
	// }
	//
	// assert.NoError(t, createColT.OnEnqueue())
	// assert.NoError(t, createColT.PreExecute(ctx))
	// assert.NoError(t, createColT.Execute(ctx))
	// assert.NoError(t, createColT.PostExecute(ctx))
	//
	// dmlChannelsFunc := getDmlChannelsFunc(ctx, rc)
	// query := newMockGetChannelsService()
	// factory := newSimpleMockMsgStreamFactory()
	//
	// collectionID, err := globalMetaCache.GetCollectionID(ctx, collectionName)
	// assert.NoError(t, err)
	//
	// qc := NewQueryCoordMock()
	// qc.Start()
	// defer qc.Stop()
	// status, err := qc.LoadCollection(ctx, &querypb.LoadCollectionRequest{
	//     Base: &commonpb.MsgBase{
	//         MsgType:   commonpb.MsgType_LoadCollection,
	//         MsgID:     0,
	//         Timestamp: 0,
	//         SourceID:  Params.ProxyCfg.ProxyID,
	//     },
	//     DbID:         0,
	//     CollectionID: collectionID,
	//     Schema:       nil,
	// })
	// assert.NoError(t, err)
	// assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
	//
	// req := constructSearchRequest("", collectionName,
	//     expr,
	//     testFloatVecField,
	//     nq, dim, nprobe, topk, roundDecimal)
	//
	// task := &searchTaskV2{
	//     Condition: NewTaskCondition(ctx),
	//     SearchRequest: &internalpb.SearchRequest{
	//         Base: &commonpb.MsgBase{
	//             MsgType:   commonpb.MsgType_Search,
	//             MsgID:     0,
	//             Timestamp: 0,
	//             SourceID:  Params.ProxyCfg.ProxyID,
	//         },
	//         ResultChannelID:    strconv.FormatInt(Params.ProxyCfg.ProxyID, 10),
	//         DbID:               0,
	//         CollectionID:       0,
	//         PartitionIDs:       nil,
	//         Dsl:                "",
	//         PlaceholderGroup:   nil,
	//         DslType:            0,
	//         SerializedExprPlan: nil,
	//         OutputFieldsId:     nil,
	//         TravelTimestamp:    0,
	//         GuaranteeTimestamp: 0,
	//     },
	//     ctx:       ctx,
	//     resultBuf: make(chan *internalpb.SearchResults, 10),
	//     result:    nil,
	//     request:   req,
	//     qc:        qc,
	//     tr:        timerecord.NewTimeRecorder("search"),
	// }
	//
	// // simple mock for query node
	// // TODO(dragondriver): should we replace this mock using RocksMq or MemMsgStream?
	//
	//
	// var wg sync.WaitGroup
	// wg.Add(1)
	// consumeCtx, cancel := context.WithCancel(ctx)
	// go func() {
	//     defer wg.Done()
	//     for {
	//         select {
	//         case <-consumeCtx.Done():
	//             return
	//         case pack, ok := <-stream.Chan():
	//             assert.True(t, ok)
	//             if pack == nil {
	//                 continue
	//             }
	//
	//             for _, msg := range pack.Msgs {
	//                 _, ok := msg.(*msgstream.SearchMsg)
	//                 assert.True(t, ok)
	//                 // TODO(dragondriver): construct result according to the request
	//
	//                 constructSearchResulstData := func() *schemapb.SearchResultData {
	//                     resultData := &schemapb.SearchResultData{
	//                         NumQueries: int64(nq),
	//                         TopK:       int64(topk),
	//                         Scores:     make([]float32, nq*topk),
	//                         Ids: &schemapb.IDs{
	//                             IdField: &schemapb.IDs_IntId{
	//                                 IntId: &schemapb.LongArray{
	//                                     Data: make([]int64, nq*topk),
	//                                 },
	//                             },
	//                         },
	//                         Topks: make([]int64, nq),
	//                     }
	//
	//                     fieldID := common.StartOfUserFieldID
	//                     for fieldName, dataType := range fieldName2Types {
	//                         resultData.FieldsData = append(resultData.FieldsData, generateFieldData(dataType, fieldName, int64(fieldID), nq*topk))
	//                         fieldID++
	//                     }
	//
	//                     for i := 0; i < nq; i++ {
	//                         for j := 0; j < topk; j++ {
	//                             offset := i*topk + j
	//                             score := float32(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()) // increasingly
	//                             id := int64(uniquegenerator.GetUniqueIntGeneratorIns().GetInt())
	//                             resultData.Scores[offset] = score
	//                             resultData.Ids.IdField.(*schemapb.IDs_IntId).IntId.Data[offset] = id
	//                         }
	//                         resultData.Topks[i] = int64(topk)
	//                     }
	//
	//                     return resultData
	//                 }
	//
	//                 result1 := &internalpb.SearchResults{
	//                     Base: &commonpb.MsgBase{
	//                         MsgType:   commonpb.MsgType_SearchResult,
	//                         MsgID:     0,
	//                         Timestamp: 0,
	//                         SourceID:  0,
	//                     },
	//                     Status: &commonpb.Status{
	//                         ErrorCode: commonpb.ErrorCode_Success,
	//                         Reason:    "",
	//                     },
	//                     ResultChannelID:          "",
	//                     MetricType:               distance.L2,
	//                     NumQueries:               int64(nq),
	//                     TopK:                     int64(topk),
	//                     SealedSegmentIDsSearched: nil,
	//                     ChannelIDsSearched:       nil,
	//                     GlobalSealedSegmentIDs:   nil,
	//                     SlicedBlob:               nil,
	//                     SlicedNumCount:           1,
	//                     SlicedOffset:             0,
	//                 }
	//                 resultData := constructSearchResulstData()
	//                 sliceBlob, err := proto.Marshal(resultData)
	//                 assert.NoError(t, err)
	//                 result1.SlicedBlob = sliceBlob
	//
	//                 // result2.SliceBlob = nil, will be skipped in decode stage
	//                 result2 := &internalpb.SearchResults{
	//                     Base: &commonpb.MsgBase{
	//                         MsgType:   commonpb.MsgType_SearchResult,
	//                         MsgID:     0,
	//                         Timestamp: 0,
	//                         SourceID:  0,
	//                     },
	//                     Status: &commonpb.Status{
	//                         ErrorCode: commonpb.ErrorCode_Success,
	//                         Reason:    "",
	//                     },
	//                     ResultChannelID:          "",
	//                     MetricType:               distance.L2,
	//                     NumQueries:               int64(nq),
	//                     TopK:                     int64(topk),
	//                     SealedSegmentIDsSearched: nil,
	//                     ChannelIDsSearched:       nil,
	//                     GlobalSealedSegmentIDs:   nil,
	//                     SlicedBlob:               nil,
	//                     SlicedNumCount:           1,
	//                     SlicedOffset:             0,
	//                 }
	//
	//                 // send search result
	//                 task.resultBuf <- result1
	//                 task.resultBuf <- result2
	//             }
	//         }
	//     }
	// }()
	//
	// assert.NoError(t, task.OnEnqueue())
	// assert.Error(t, task.PreExecute(ctx))
	//
	// cancel()
	// wg.Wait()
}

func TestSearchTaskV2_all(t *testing.T) {
	// var err error
	//
	// Params.Init()
	// Params.ProxyCfg.SearchResultChannelNames = []string{funcutil.GenRandomStr()}
	//
	// rc := NewRootCoordMock()
	// rc.Start()
	// defer rc.Stop()
	//
	// ctx := context.Background()
	//
	// err = InitMetaCache(rc)
	// assert.NoError(t, err)
	//
	// shardsNum := int32(2)
	// prefix := "TestSearchTaskV2_all"
	// collectionName := prefix + funcutil.GenRandomStr()
	//
	// dim := 128
	// expr := fmt.Sprintf("%s > 0", testInt64Field)
	// nq := 10
	// topk := 10
	// roundDecimal := 3
	// nprobe := 10
	//
	// fieldName2Types := map[string]schemapb.DataType{
	//     testBoolField:     schemapb.DataType_Bool,
	//     testInt32Field:    schemapb.DataType_Int32,
	//     testInt64Field:    schemapb.DataType_Int64,
	//     testFloatField:    schemapb.DataType_Float,
	//     testDoubleField:   schemapb.DataType_Double,
	//     testFloatVecField: schemapb.DataType_FloatVector,
	// }
	// if enableMultipleVectorFields {
	//     fieldName2Types[testBinaryVecField] = schemapb.DataType_BinaryVector
	// }
	//
	// schema := constructCollectionSchemaByDataType(collectionName, fieldName2Types, testInt64Field, false)
	// marshaledSchema, err := proto.Marshal(schema)
	// assert.NoError(t, err)
	//
	// createColT := &createCollectionTask{
	//     Condition: NewTaskCondition(ctx),
	//     CreateCollectionRequest: &milvuspb.CreateCollectionRequest{
	//         Base:           nil,
	//         CollectionName: collectionName,
	//         Schema:         marshaledSchema,
	//         ShardsNum:      shardsNum,
	//     },
	//     ctx:       ctx,
	//     rootCoord: rc,
	//     result:    nil,
	//     schema:    nil,
	// }
	//
	// assert.NoError(t, createColT.OnEnqueue())
	// assert.NoError(t, createColT.PreExecute(ctx))
	// assert.NoError(t, createColT.Execute(ctx))
	// assert.NoError(t, createColT.PostExecute(ctx))
	//
	// dmlChannelsFunc := getDmlChannelsFunc(ctx, rc)
	// query := newMockGetChannelsService()
	// factory := newSimpleMockMsgStreamFactory()
	//
	// collectionID, err := globalMetaCache.GetCollectionID(ctx, collectionName)
	// assert.NoError(t, err)
	//
	// qc := NewQueryCoordMock()
	// qc.Start()
	// defer qc.Stop()
	// status, err := qc.LoadCollection(ctx, &querypb.LoadCollectionRequest{
	//     Base: &commonpb.MsgBase{
	//         MsgType:   commonpb.MsgType_LoadCollection,
	//         MsgID:     0,
	//         Timestamp: 0,
	//         SourceID:  Params.ProxyCfg.ProxyID,
	//     },
	//     DbID:         0,
	//     CollectionID: collectionID,
	//     Schema:       nil,
	// })
	// assert.NoError(t, err)
	// assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
	//
	// req := constructSearchRequest("", collectionName,
	//     expr,
	//     testFloatVecField,
	//     nq, dim, nprobe, topk, roundDecimal)
	//
	// task := &searchTaskV2{
	//     Condition: NewTaskCondition(ctx),
	//     SearchRequest: &internalpb.SearchRequest{
	//         Base: &commonpb.MsgBase{
	//             MsgType:   commonpb.MsgType_Search,
	//             MsgID:     0,
	//             Timestamp: 0,
	//             SourceID:  Params.ProxyCfg.ProxyID,
	//         },
	//         ResultChannelID:    strconv.FormatInt(Params.ProxyCfg.ProxyID, 10),
	//         DbID:               0,
	//         CollectionID:       0,
	//         PartitionIDs:       nil,
	//         Dsl:                "",
	//         PlaceholderGroup:   nil,
	//         DslType:            0,
	//         SerializedExprPlan: nil,
	//         OutputFieldsId:     nil,
	//         TravelTimestamp:    0,
	//         GuaranteeTimestamp: 0,
	//     },
	//     ctx:       ctx,
	//     resultBuf: make(chan *internalpb.SearchResults, 10),
	//     result:    nil,
	//     request:   req,
	//     qc:        qc,
	//     tr:        timerecord.NewTimeRecorder("search"),
	// }
	//
	// // simple mock for query node
	// // TODO(dragondriver): should we replace this mock using RocksMq or MemMsgStream?
	//
	// var wg sync.WaitGroup
	// wg.Add(1)
	// consumeCtx, cancel := context.WithCancel(ctx)
	// go func() {
	//     defer wg.Done()
	//     for {
	//         select {
	//         case <-consumeCtx.Done():
	//             return
	//         case pack, ok := <-stream.Chan():
	//             assert.True(t, ok)
	//             if pack == nil {
	//                 continue
	//             }
	//
	//             for _, msg := range pack.Msgs {
	//                 _, ok := msg.(*msgstream.SearchMsg)
	//                 assert.True(t, ok)
	//                 // TODO(dragondriver): construct result according to the request
	//
	//                 constructSearchResulstData := func() *schemapb.SearchResultData {
	//                     resultData := &schemapb.SearchResultData{
	//                         NumQueries: int64(nq),
	//                         TopK:       int64(topk),
	//                         Scores:     make([]float32, nq*topk),
	//                         Ids: &schemapb.IDs{
	//                             IdField: &schemapb.IDs_IntId{
	//                                 IntId: &schemapb.LongArray{
	//                                     Data: make([]int64, nq*topk),
	//                                 },
	//                             },
	//                         },
	//                         Topks: make([]int64, nq),
	//                     }
	//
	//                     fieldID := common.StartOfUserFieldID
	//                     for fieldName, dataType := range fieldName2Types {
	//                         resultData.FieldsData = append(resultData.FieldsData, generateFieldData(dataType, fieldName, int64(fieldID), nq*topk))
	//                         fieldID++
	//                     }
	//
	//                     for i := 0; i < nq; i++ {
	//                         for j := 0; j < topk; j++ {
	//                             offset := i*topk + j
	//                             score := float32(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()) // increasingly
	//                             id := int64(uniquegenerator.GetUniqueIntGeneratorIns().GetInt())
	//                             resultData.Scores[offset] = score
	//                             resultData.Ids.IdField.(*schemapb.IDs_IntId).IntId.Data[offset] = id
	//                         }
	//                         resultData.Topks[i] = int64(topk)
	//                     }
	//
	//                     return resultData
	//                 }
	//
	//                 result1 := &internalpb.SearchResults{
	//                     Base: &commonpb.MsgBase{
	//                         MsgType:   commonpb.MsgType_SearchResult,
	//                         MsgID:     0,
	//                         Timestamp: 0,
	//                         SourceID:  0,
	//                     },
	//                     Status: &commonpb.Status{
	//                         ErrorCode: commonpb.ErrorCode_Success,
	//                         Reason:    "",
	//                     },
	//                     ResultChannelID:          "",
	//                     MetricType:               distance.L2,
	//                     NumQueries:               int64(nq),
	//                     TopK:                     int64(topk),
	//                     SealedSegmentIDsSearched: nil,
	//                     ChannelIDsSearched:       nil,
	//                     GlobalSealedSegmentIDs:   nil,
	//                     SlicedBlob:               nil,
	//                     SlicedNumCount:           1,
	//                     SlicedOffset:             0,
	//                 }
	//                 resultData := constructSearchResulstData()
	//                 sliceBlob, err := proto.Marshal(resultData)
	//                 assert.NoError(t, err)
	//                 result1.SlicedBlob = sliceBlob
	//
	//                 // result2.SliceBlob = nil, will be skipped in decode stage
	//                 result2 := &internalpb.SearchResults{
	//                     Base: &commonpb.MsgBase{
	//                         MsgType:   commonpb.MsgType_SearchResult,
	//                         MsgID:     0,
	//                         Timestamp: 0,
	//                         SourceID:  0,
	//                     },
	//                     Status: &commonpb.Status{
	//                         ErrorCode: commonpb.ErrorCode_Success,
	//                         Reason:    "",
	//                     },
	//                     ResultChannelID:          "",
	//                     MetricType:               distance.L2,
	//                     NumQueries:               int64(nq),
	//                     TopK:                     int64(topk),
	//                     SealedSegmentIDsSearched: nil,
	//                     ChannelIDsSearched:       nil,
	//                     GlobalSealedSegmentIDs:   nil,
	//                     SlicedBlob:               nil,
	//                     SlicedNumCount:           1,
	//                     SlicedOffset:             0,
	//                 }
	//
	//                 // send search result
	//                 task.resultBuf <- result1
	//                 task.resultBuf <- result2
	//             }
	//         }
	//     }
	// }()
	//
	// assert.NoError(t, task.OnEnqueue())
	// assert.NoError(t, task.PreExecute(ctx))
	// assert.NoError(t, task.Execute(ctx))
	// assert.NoError(t, task.PostExecute(ctx))
	//
	// cancel()
	// wg.Wait()
}

func TestSearchTaskV2_7803_reduce(t *testing.T) {
	// var err error
	//
	// Params.Init()
	// Params.ProxyCfg.SearchResultChannelNames = []string{funcutil.GenRandomStr()}
	//
	// rc := NewRootCoordMock()
	// rc.Start()
	// defer rc.Stop()
	//
	// ctx := context.Background()
	//
	// err = InitMetaCache(rc)
	// assert.NoError(t, err)
	//
	// shardsNum := int32(2)
	// prefix := "TestSearchTaskV2_7803_reduce"
	// collectionName := prefix + funcutil.GenRandomStr()
	// int64Field := "int64"
	// floatVecField := "fvec"
	// dim := 128
	// expr := fmt.Sprintf("%s > 0", int64Field)
	// nq := 10
	// topk := 10
	// roundDecimal := 3
	// nprobe := 10
	//
	// schema := constructCollectionSchema(
	//     int64Field,
	//     floatVecField,
	//     dim,
	//     collectionName)
	// marshaledSchema, err := proto.Marshal(schema)
	// assert.NoError(t, err)
	//
	// createColT := &createCollectionTask{
	//     Condition: NewTaskCondition(ctx),
	//     CreateCollectionRequest: &milvuspb.CreateCollectionRequest{
	//         Base:           nil,
	//         CollectionName: collectionName,
	//         Schema:         marshaledSchema,
	//         ShardsNum:      shardsNum,
	//     },
	//     ctx:       ctx,
	//     rootCoord: rc,
	//     result:    nil,
	//     schema:    nil,
	// }
	//
	// assert.NoError(t, createColT.OnEnqueue())
	// assert.NoError(t, createColT.PreExecute(ctx))
	// assert.NoError(t, createColT.Execute(ctx))
	// assert.NoError(t, createColT.PostExecute(ctx))
	//
	// dmlChannelsFunc := getDmlChannelsFunc(ctx, rc)
	// query := newMockGetChannelsService()
	// factory := newSimpleMockMsgStreamFactory()
	//
	// collectionID, err := globalMetaCache.GetCollectionID(ctx, collectionName)
	// assert.NoError(t, err)
	//
	// qc := NewQueryCoordMock()
	// qc.Start()
	// defer qc.Stop()
	// status, err := qc.LoadCollection(ctx, &querypb.LoadCollectionRequest{
	//     Base: &commonpb.MsgBase{
	//         MsgType:   commonpb.MsgType_LoadCollection,
	//         MsgID:     0,
	//         Timestamp: 0,
	//         SourceID:  Params.ProxyCfg.ProxyID,
	//     },
	//     DbID:         0,
	//     CollectionID: collectionID,
	//     Schema:       nil,
	// })
	// assert.NoError(t, err)
	// assert.Equal(t, commonpb.ErrorCode_Success, status.ErrorCode)
	//
	// req := constructSearchRequest("", collectionName,
	//     expr,
	//     floatVecField,
	//     nq, dim, nprobe, topk, roundDecimal)
	//
	// task := &searchTaskV2{
	//     Condition: NewTaskCondition(ctx),
	//     SearchRequest: &internalpb.SearchRequest{
	//         Base: &commonpb.MsgBase{
	//             MsgType:   commonpb.MsgType_Search,
	//             MsgID:     0,
	//             Timestamp: 0,
	//             SourceID:  Params.ProxyCfg.ProxyID,
	//         },
	//         ResultChannelID:    strconv.FormatInt(Params.ProxyCfg.ProxyID, 10),
	//         DbID:               0,
	//         CollectionID:       0,
	//         PartitionIDs:       nil,
	//         Dsl:                "",
	//         PlaceholderGroup:   nil,
	//         DslType:            0,
	//         SerializedExprPlan: nil,
	//         OutputFieldsId:     nil,
	//         TravelTimestamp:    0,
	//         GuaranteeTimestamp: 0,
	//     },
	//     ctx:       ctx,
	//     resultBuf: make(chan *internalpb.SearchResults, 10),
	//     result:    nil,
	//     request:   req,
	//     qc:        qc,
	//     tr:        timerecord.NewTimeRecorder("search"),
	// }
	//
	// // simple mock for query node
	// // TODO(dragondriver): should we replace this mock using RocksMq or MemMsgStream?
	//
	// var wg sync.WaitGroup
	// wg.Add(1)
	// consumeCtx, cancel := context.WithCancel(ctx)
	// go func() {
	//     defer wg.Done()
	//     for {
	//         select {
	//         case <-consumeCtx.Done():
	//             return
	//         case pack, ok := <-stream.Chan():
	//             assert.True(t, ok)
	//             if pack == nil {
	//                 continue
	//             }
	//
	//             for _, msg := range pack.Msgs {
	//                 _, ok := msg.(*msgstream.SearchMsg)
	//                 assert.True(t, ok)
	//                 // TODO(dragondriver): construct result according to the request
	//
	//                 constructSearchResulstData := func(invalidNum int) *schemapb.SearchResultData {
	//                     resultData := &schemapb.SearchResultData{
	//                         NumQueries: int64(nq),
	//                         TopK:       int64(topk),
	//                         FieldsData: nil,
	//                         Scores:     make([]float32, nq*topk),
	//                         Ids: &schemapb.IDs{
	//                             IdField: &schemapb.IDs_IntId{
	//                                 IntId: &schemapb.LongArray{
	//                                     Data: make([]int64, nq*topk),
	//                                 },
	//                             },
	//                         },
	//                         Topks: make([]int64, nq),
	//                     }
	//
	//                     for i := 0; i < nq; i++ {
	//                         for j := 0; j < topk; j++ {
	//                             offset := i*topk + j
	//                             if j >= invalidNum {
	//                                 resultData.Scores[offset] = minFloat32
	//                                 resultData.Ids.IdField.(*schemapb.IDs_IntId).IntId.Data[offset] = -1
	//                             } else {
	//                                 score := float32(uniquegenerator.GetUniqueIntGeneratorIns().GetInt()) // increasingly
	//                                 id := int64(uniquegenerator.GetUniqueIntGeneratorIns().GetInt())
	//                                 resultData.Scores[offset] = score
	//                                 resultData.Ids.IdField.(*schemapb.IDs_IntId).IntId.Data[offset] = id
	//                             }
	//                         }
	//                         resultData.Topks[i] = int64(topk)
	//                     }
	//
	//                     return resultData
	//                 }
	//
	//                 result1 := &internalpb.SearchResults{
	//                     Base: &commonpb.MsgBase{
	//                         MsgType:   commonpb.MsgType_SearchResult,
	//                         MsgID:     0,
	//                         Timestamp: 0,
	//                         SourceID:  0,
	//                     },
	//                     Status: &commonpb.Status{
	//                         ErrorCode: commonpb.ErrorCode_Success,
	//                         Reason:    "",
	//                     },
	//                     ResultChannelID:          "",
	//                     MetricType:               distance.L2,
	//                     NumQueries:               int64(nq),
	//                     TopK:                     int64(topk),
	//                     SealedSegmentIDsSearched: nil,
	//                     ChannelIDsSearched:       nil,
	//                     GlobalSealedSegmentIDs:   nil,
	//                     SlicedBlob:               nil,
	//                     SlicedNumCount:           1,
	//                     SlicedOffset:             0,
	//                 }
	//                 resultData := constructSearchResulstData(topk / 2)
	//                 sliceBlob, err := proto.Marshal(resultData)
	//                 assert.NoError(t, err)
	//                 result1.SlicedBlob = sliceBlob
	//
	//                 result2 := &internalpb.SearchResults{
	//                     Base: &commonpb.MsgBase{
	//                         MsgType:   commonpb.MsgType_SearchResult,
	//                         MsgID:     0,
	//                         Timestamp: 0,
	//                         SourceID:  0,
	//                     },
	//                     Status: &commonpb.Status{
	//                         ErrorCode: commonpb.ErrorCode_Success,
	//                         Reason:    "",
	//                     },
	//                     ResultChannelID:          "",
	//                     MetricType:               distance.L2,
	//                     NumQueries:               int64(nq),
	//                     TopK:                     int64(topk),
	//                     SealedSegmentIDsSearched: nil,
	//                     ChannelIDsSearched:       nil,
	//                     GlobalSealedSegmentIDs:   nil,
	//                     SlicedBlob:               nil,
	//                     SlicedNumCount:           1,
	//                     SlicedOffset:             0,
	//                 }
	//                 resultData2 := constructSearchResulstData(topk - topk/2)
	//                 sliceBlob2, err := proto.Marshal(resultData2)
	//                 assert.NoError(t, err)
	//                 result2.SlicedBlob = sliceBlob2
	//
	//                 // send search result
	//                 task.resultBuf <- result1
	//                 task.resultBuf <- result2
	//             }
	//         }
	//     }
	// }()
	//
	// assert.NoError(t, task.OnEnqueue())
	// assert.NoError(t, task.PreExecute(ctx))
	// assert.NoError(t, task.Execute(ctx))
	// assert.NoError(t, task.PostExecute(ctx))
	//
	// cancel()
	// wg.Wait()
}
