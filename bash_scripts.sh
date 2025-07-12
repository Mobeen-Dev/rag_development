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

# Image Startup 
    sudo docker run -p 6333:6333 -p 6334:6334  -v "./DB:/qdrant/storage:z"  qdrant/qdrant
    sudo docker run -d -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin"  -v opensearch-data:/usr/share/opensearch/data opensearchproject/opensearch:latest # id:pass = admin:admin (Development Only)