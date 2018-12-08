
import numpy as np
import tensorflow as tf


class BatchData():
    def __init__(self):
        self.input_data = None
        self.input_len = None
        self.label = None


class GetBatchData():
    def __init__(self, input_width, input_height, batch_size):
        self.input_width = input_width
        self.input_heigth = input_height
        self.batch_size = batch_size

    def getBatch(self):
        baths = []
        for _ in range(10):
            input_len = []
            batch_datas = []
            input_labesl = []
            for _ in range(self.batch_size):
                data = []
                for i in range(self.input_heigth):
                    ele = np.random.randint(low=1,high=10,size=self.input_width).tolist()
                    data.append(ele)
                ele = np.random.randint(low=1,high=10,size=self.input_heigth).tolist()
                input_labesl.append(ele)
                input_len.append(self.input_heigth)
                batch_datas.append(data)
            batch = BatchData()
            batch.input_len= input_len
            batch.label = input_labesl
            batch.input_data = batch_datas
            baths.append(batch)
        return baths





if __name__ == '__main__':
    fn = GetBatchData(10,10,5)
    from stock_lstm_model import StockModel
    model = StockModel(200,3,10,10,1000)
    batches = fn.getBatch()
    model.train(batches)

