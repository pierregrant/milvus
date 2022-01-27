package main

import (
	"bytes"
	"flag"
	"fmt"

	"github.com/golang/protobuf/proto"
	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/proto/etcdpb"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"go.uber.org/zap"
)

var (
	etcdAddr = flag.String("etcd", "127.0.0.1:2379", "Etcd Endpoint to connect")
	rootPath = flag.String("rootPath", "by-dev/meta/root-coord/collection/", "Root Coord collection root path to iterate")
)

func main() {
	flag.Parse()

	etcdCli, err := etcd.GetRemoteEtcdClient([]string{*etcdAddr})
	if err != nil {
		log.Fatal("failed to connect to etcd server", zap.Error(err))
	}
	etcdkv := etcdkv.NewEtcdKV(etcdCli, *rootPath)

	keys, values, err := etcdkv.LoadWithPrefix("/")
	if err != nil {
		log.Fatal("failed to list ", zap.Error(err))
	}
	living := 0
	channels := 0
	for i := range keys {
		if bytes.Equal([]byte(values[i]), []byte{0xE2, 0x9B, 0xBC}) {
			continue
		}
		info := &etcdpb.CollectionInfo{}
		err = proto.Unmarshal([]byte(values[i]), info)
		if err != nil {
			continue
		}

		living++
		printColletionInfo(info)
		channels += len(info.VirtualChannelNames)
	}
	fmt.Printf("Remaining collection: %d, virtual channels: %d\n", living, channels)
}

const (
	tsPrintFormat = "2006-01-02 15:04:05.999 -0700"
)

func printColletionInfo(info *etcdpb.CollectionInfo) {
	fmt.Println("================================================================================")
	fmt.Printf("Collection ID: %d\n", info.ID)
	fmt.Printf("Collection Name :%s\n", info.GetSchema().GetName())

	fmt.Println("================================================================================")
}
