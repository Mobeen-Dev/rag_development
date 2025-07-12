# Prerequisites:
    # Disable memory paging and swapping performance on the host to improve performance.
    sudo swapoff -a

    sudo nano /etc/sysctl.conf
    vm.max_map_count=262144
    sudo sysctl -p
    cat /sys/vm/max_map_count


# Image Required
    sudo docker pull qdrant/qdrant
    sudo docker pull opensearchproject/opensearch:3
    sudo docker pull opensearchproject/opensearch-dashboards:3

sudo docker run -p 6333:6333 -p 6334:6334  -v "./DB:/qdrant/storage:z"  qdrant/qdrant