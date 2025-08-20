SH=`cd $(dirname ${BASH_SOURCE:-$0}) && pwd`
cd $SH
docker compose up --build -d

echo
echo "cd $SH"
echo "docker compose logs -f"
