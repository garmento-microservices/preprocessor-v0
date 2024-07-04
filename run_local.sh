container=$(docker run -e MYSQL_ROOT_PASSWORD=1 -e MYSQL_DATABASE=preprocessing -p 6603:3306 --rm -d mysql:8.4)
export DB_CONNECTION_STR="mysql://root:1@127.0.0.1:6603/preprocessing"
echo "wait for DB to be online"
sleep 10
pdm run apply_migration
pdm run dev
docker stop $container
