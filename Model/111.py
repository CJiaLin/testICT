from Parm import GetInf
#import ssl
#import gevent
import time
# import gevent.monkey
# gevent.monkey.patch_all()
# import warnings
#warnings.filterwarnings("ignore")

db = GetInf()

patent,session = db.ENbyDB()