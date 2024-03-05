import click
from controller import main
import time

@click.command()
@click.option('-s', '--my_data')
def start_pipeline(my_data):
    print("모델 예측 시작!!!")
        
    main(my_data)

    time.sleep(2)




if __name__ == '__main__':
    start_pipeline()