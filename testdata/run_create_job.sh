# curl -X "POST" \
#     -H "Content-Type: application/json" \
#     -d @./test_data.json http://localhost:8000/jobs
curl -X "POST" \
    -H "Content-Type: multipart/form-data" \
    -F "ref_image=@origin.jpg" \
    -F "garment_image=@garment.jpg" \
    http://localhost:8000/jobs
