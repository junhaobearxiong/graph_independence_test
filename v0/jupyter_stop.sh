if [ ! -e .jupyter_pid ]
then
    echo "No stored jupyter PID"
    exit 1
fi

kill $(cat .jupyter_pid)
rm -f .jupyter_pid
