PYTHONPATH=/Users/jxiong/Documents/Projects/:$(pwd) jupyter notebook > /tmp/jupyter.log 2>&1 &
echo $! > .jupyter_pid
