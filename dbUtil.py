import configparser
import pymysql
import datetime

from predict import CnnModel

BATCH_SIZE = 100


class CorpusSQL:
    def __init__(self, host, user, password, db):
        self.mysqlConn = pymysql.connect(host=host,
                                         user=user,
                                         password=password,
                                         db=db,
                                         charset='utf8mb4',
                                         cursorclass=pymysql.cursors.DictCursor)

    def selectDistinctEvents(self):
        sqlStr = """
        SELECT DISTINCT `event` from `weibo_origin`;
        """
        with self.mysqlConn.cursor() as cs:
            try:
                cs.execute(sqlStr)
                events = cs.fetchall()
                return [event['event'] for event in events]
            except Exception as e:
                print(e)

    def selectWeiboOriginByEvent(self, eventName):
        sqlStr = """
        SELECT `rowkey`, `id`, `content`, `event` FROM weibo_origin WHERE
        ISNULL(`type`) AND `event` = %s
        """
        with self.mysqlConn.cursor() as cs:
            try:
                cs.execute(sqlStr, eventName)
                contents = cs.fetchall()
                return contents
            except Exception as e:
                print(e)

    def update(self, fieldList):
        """
        fieldTuple:
                [(type, prob, rowkey), (type, prob, rowkey), ..., (type, prob, rowkey)]
        """
        sqlStr = """
        UPDATE `weibo_origin` SET `type` = %s, `prob` = %s WHERE `rowkey` = %s
        """
        with self.mysqlConn.cursor() as cs:
            try:
                cs.executemany(sqlStr, fieldList)
                self.mysqlConn.commit()
                # print("%s UPDATE SUCCESS!" % args[2])
            except Exception as e:
                print(e)
                self.mysqlConn.rollback()

    def __del__(self):
        self.mysqlConn.close()


def corpusPredictSinglePiece(model, rowkey, corpusContent):
    probabilities, label = model.predict(corpusContent)
    # print("Processing ROWKEY: [%s]" % str(rowkeys))
    return probabilities, label


def corpusPredict(model, rowkeys, corpusContents):
    probabilities, label = model.predictBatch(corpusContents)
    # print("Processing ROWKEY: [%s]" % str(rowkeys))
    return probabilities, label


def batchUpdate(sqlConn, updateFields):
    """
    对数据库进行批处理更新，加快更新速度，批处理大小自由
    或者是BATCH_SIZE，或者是剩余的余数大小
    """
    print("=" * 100 + "\n\t\t\t\t\tUPDATE\n" + "=" * 100)
    sqlConn.update(updateFields)


def sqlConfigRead():
    cf = configparser.ConfigParser()
    cf.read("config.ini")
    host = cf.get("MySQL", "HOST")
    user = cf.get("MySQL", "USER")
    password = cf.get("MySQL", "PASSWORD")
    db = cf.get("MySQL", "DB")
    print(host, user, password, db)
    return host, user, password, db


def main():
    host, user, password, db = sqlConfigRead()
    start = datetime.datetime.now()
    cnn_model = CnnModel()
    sqlConn = CorpusSQL(host, user, password, db)
    eventList = sqlConn.selectDistinctEvents()
    for eventExample in eventList:
        eventCorpora = sqlConn.selectWeiboOriginByEvent(eventExample)
        eventLen = len(eventCorpora)
        print("EVENT [%s] Contains %s ITEMS" % (eventExample, eventLen))

        updateFields = []
        if eventLen <= BATCH_SIZE:
            for eventCorpus in eventCorpora:
                rowkey = eventCorpus['rowkey']
                content = eventCorpus['content']
                probs, label = corpusPredictSinglePiece(cnn_model, rowkey, content)
                # ndArray数组维度为2，去掉最外层中括号
                probDistribution = probs.tolist()[0]
                updateFields.append((label, str(probDistribution), rowkey))
            batchUpdate(sqlConn, updateFields)
            continue

        for iter in range(int(eventLen / BATCH_SIZE) + 1):
            startIndex = iter * BATCH_SIZE
            endIndex = (iter + 1) * BATCH_SIZE
            if endIndex >= eventLen:
                endIndex = eventLen
            # 处理从startIndex 到 endIndex 范围内的语料
            updateFields.clear()
            rowkeys = [eventCorpus['rowkey'] for eventCorpus in eventCorpora[startIndex: endIndex]]
            contents = [eventCorpus['content'] for eventCorpus in eventCorpora[startIndex: endIndex]]
            probsList, labels = corpusPredict(cnn_model, rowkeys, contents)
            for probs, label, rowkey in zip(probsList, labels, rowkeys):
                # ndArray数组维度为2，去掉最外层中括号
                probDistribution = probs.tolist()[0]
                updateFields.append((label, str(probDistribution), rowkey))
            batchUpdate(sqlConn, updateFields)

    end = datetime.datetime.now()
    print("Start Time: %s\nEnd Time: %s\nPrediction During Time: %.3ss" % (start, end, (end - start).seconds))
    print("PROCESS COMPLETE.")


if __name__ == '__main__':
    main()
