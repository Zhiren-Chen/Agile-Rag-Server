import uvicorn
import services
from configs import HOST, PORT
from handler.manager import observer

if __name__=='__main__':
    uvicorn.run(app="services:app", host=HOST, port=int(PORT), reload=False)
    try:
        i=0
        while True:
            time.sleep(0.4)  
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
#python main.py