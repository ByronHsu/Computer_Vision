echo "Download mono models"
wget 'https://www.dropbox.com/s/8757a5vjbrfv437/model-200.data-00000-of-00001?dl=1' -O model/mono/checkpoints/model-200.data-00000-of-00001
wget 'https://www.dropbox.com/s/4pavm79kmaaprng/model-200.index?dl=1' -O model/mono/checkpoints/model-200.index
wget 'https://www.dropbox.com/s/l0xnijf8dtujihj/model-200.meta?dl=1' -O model/mono/checkpoints/model-200.meta

echo "Download classifier models"
wget 'https://www.dropbox.com/s/7oc5w5d31lomzt7/pretrain.pt?dl=1' -O model/classifier/checkpoints/pretrain.pt
