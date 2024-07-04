TAG="garmento.io/preprocessor"
CLOTH_SEG_WEIGHTS_PATH="tmp/weights.zip"
CLOTH_SEG_WEIGHTS_URL="https://github.com/ternaus/cloths_segmentation/releases/download/0.0.1/weights.zip"

if ! [ -f ${CLOTH_SEG_WEIGHTS_PATH} ]; then
    mkdir -p tmp
    wget -O ${CLOTH_SEG_WEIGHTS_PATH} ${CLOTH_SEG_WEIGHTS_URL}
fi

docker build -t $TAG .
