SH=`cd $(dirname ${BASH_SOURCE:-$0}) && pwd`

python -m pipenv run \
  python "$SH/../app_test.py"
