package querynode

import (
	"context"
	"fmt"
	"path"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/milvus-io/milvus/internal/proto/querypb"
	"github.com/milvus-io/milvus/internal/util/funcutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3client"
)

func TestEtcdShardSegmentDetector_watch(t *testing.T) {
	etcdServer, err := startEmbedEtcdServer()
	require.NoError(t, err)
	defer etcdServer.Close()

	client := v3client.New(etcdServer.Server)
	defer client.Close()

	type testCase struct {
		name               string
		oldRecords         map[string]*querypb.SegmentInfo
		updateRecords      map[string]*querypb.SegmentInfo
		delRecords         []string
		expectInitEvents   []segmentEvent
		expectUpdateEvents []segmentEvent

		collectionID int64
		replicaID    int64
		channel      string
	}
	cases := []testCase{
		{
			name:         "normal init",
			collectionID: 1,
			replicaID:    1,
			channel:      "dml_1_1_v0",
			oldRecords: map[string]*querypb.SegmentInfo{
				"segment_1": {
					CollectionID: 1,
					SegmentID:    1,
					NodeID:       1,
					DmChannel:    "dml_1_1_v0",
					ReplicaIds:   []int64{1, 2},
				},
			},
			expectInitEvents: []segmentEvent{
				{
					eventType: segmentAdd,
					segmentID: 1,
					nodeID:    1,
					state:     segmentStateLoading,
				},
			},
		},
		{
			name:         "normal init with other segments",
			collectionID: 1,
			replicaID:    1,
			channel:      "dml_1_1_v0",
			oldRecords: map[string]*querypb.SegmentInfo{
				"segment_1": {
					CollectionID: 1,
					SegmentID:    1,
					NodeID:       1,
					DmChannel:    "dml_1_1_v0",
					ReplicaIds:   []int64{1, 2},
				},
				"segment_2": {
					CollectionID: 1,
					SegmentID:    2,
					NodeID:       1,
					DmChannel:    "dml_1_1_v1",
					ReplicaIds:   []int64{1, 2},
				},
				"segment_3": {
					CollectionID: 2,
					SegmentID:    3,
					NodeID:       2,
					DmChannel:    "dml_3_2_v0",
					ReplicaIds:   []int64{1, 2},
				},
				"segment_4": { // may not happen
					CollectionID: 1,
					SegmentID:    4,
					NodeID:       1,
					DmChannel:    "dml_1_1_v0",
					ReplicaIds:   []int64{2},
				},
			},
			expectInitEvents: []segmentEvent{
				{
					eventType: segmentAdd,
					segmentID: 1,
					nodeID:    1,
					state:     segmentStateLoading,
				},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			suffix := funcutil.RandomString(6)
			rootPath := fmt.Sprintf("qn_shard_segment_detector_watch_%s", suffix)

			ctx := context.Background()
			// put existing records
			for key, info := range tc.oldRecords {
				bs, err := proto.Marshal(info)
				require.NoError(t, err)
				_, err = client.Put(ctx, path.Join(rootPath, key), string(bs))
				require.NoError(t, err)
			}

			sd := NewEtcdShardSegmentDetector(client, rootPath)
			segments, ch := sd.watchSegments(tc.collectionID, tc.replicaID, tc.channel)
			assert.ElementsMatch(t, tc.expectInitEvents, segments)

			// update etcd kvs to generate events
			go func() {
				for key, info := range tc.updateRecords {
					bs, err := proto.Marshal(info)
					require.NoError(t, err)
					_, err = client.Put(ctx, path.Join(rootPath, key), string(bs))
					require.NoError(t, err)
				}
				for _, k := range tc.delRecords {
					_, err := client.Delete(ctx, path.Join(rootPath, k))
					require.NoError(t, err)
				}
				// need a way to detect event processed
				time.Sleep(time.Millisecond * 100)

				sd.Close()
			}()

			var newEvents []segmentEvent
			for evt := range ch {
				newEvents = append(newEvents, evt)
			}
			assert.ElementsMatch(t, tc.expectUpdateEvents, newEvents)
		})
	}
}
