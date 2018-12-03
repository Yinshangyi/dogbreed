bind = "127.0.0.1:9000"                   # Don't use port 80 becaue nginx occupied it already. 
errorlog = '/Users/yinshangyi/Documents/ML-Projects/API-Job/logs/gunicorn-error.log'  # Make sure you have the log folder create
accesslog = '/Users/yinshangyi/Documents/ML-Projects/API-Job/logs/gunicorn-access.log'
loglevel = 'debug'
timeout = 120
workers = 1     # the number of recommended workers is '2 * number of CPUs + 1' 