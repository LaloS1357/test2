                                #!/bin/bash
cd ~/test2 || exit
git pull
docker-compose down
docker-compose up --build -d
docker-compose logs -f --tail=10