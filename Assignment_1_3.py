from __future__ import print_function
from __future__ import division
from pyspark import SparkConf, SparkContext
from datetime import datetime
import os, sys, logging, mmh3, math
from optparse import OptionParser


DIRECTORY_TO_SAVE = "/Users/clonyjr/Library/Mobile Documents/com~apple~CloudDocs/" \
                    "Aveiro/UA/CLONY/MEI/2019-2020-SEM-2/MDLE/Assignments/Assignment 1/MovieSummaries"
# "s3://clony94085-assignment1/data"
FILE_NAME_TO_READ = "plot_summaries_sample2.txt"

FILEPATH = os.path.join(DIRECTORY_TO_SAVE, FILE_NAME_TO_READ)

logging.basicConfig(filename='lshlog.log', filemode='a', format='%(asctime)s %(levelname)s - %(message)s'
                    , datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

APP_NAME = 'LSH'
BANDS = 20
ROWS = 5 # values in each Band

PRINT_TIME = True
OUTPUT_FILE = "output.txt"


def setOutputData(filename='', jaccard_similarity_dict={}):
    # output results.
    try:
        if filename is not None:
            orig_stdout = sys.stdout
            f = open(filename, 'w')
            sys.stdout = f
        else:
            pass
        ##########

        logging.debug('**jaccard_similarity_dict = %s' % (jaccard_similarity_dict))
        for item in jaccard_similarity_dict:
            print("Key : {} , Value : {}".format(item, jaccard_similarity_dict[item]))

        ###########
        sys.stdout.flush()
        if filename != None :
            sys.stdout = orig_stdout
            f.close()
        else:
            pass
    except IOError as _err:
        logging.error('File error: ' + str(_err))
        exit()
    logging.info('Locality Sensitive Hashing =>Finish=>%s' % (str(datetime.now())))


def schingling(doc):
    return [mmh3.hash(doc[i:i+9], signed=False)for i in range(len(doc)-9)]


def customized_hash(data, seed):
    '''
    This function implements a customized hash function for Minhashing.
        Data = a data/item to get hash value.
        Seed = a seed number to generate different hash function.
    '''
    return (3*int(data) + 13*int(seed)) % 100


class Minhashing(object):
    '''
    This class implements Minhashing algorithm.
        Calculate signature (hash value) for each item in the specified file.
        Return the minimum number of hash values
    '''
    hash_func = None

    def __init__(self, hash_function= customized_hash):
        '''
        Constructor
        '''
        Minhashing.hash_func = staticmethod(hash_function)

    @staticmethod
    def get_value(data_list=[], seed=0):
        _signatures = []
        _signature = None

        logging.debug('Minhash.get_signature=>data_list=%s, seed=%d' % (data_list, seed))
        for data in data_list:
            _signature = Minhashing.hash_func(data, seed)
            _signatures.append(_signature)
        logging.debug('Minhash.get_signature=>_signatures=%s'%(_signatures))
        return min(_signatures)

class LSH(object):
    bands = None
    rows = None
    hash_alg = None
    jaccard_similarity = {} # {(set1, set2): jaccard_similarity, ...}

    def __init__(self, bands=BANDS, rows = ROWS, hash_function=customized_hash):

        self.conf = SparkConf().setAppName(APP_NAME).setMaster("local[*]")
        # Create a context for the job.
        self.sc = SparkContext(conf=self.conf)
        logging.debug("Apache-Spark started")

        data = self.sc.textFile(FILEPATH).map(lambda l: l.split("\t"))
        data_set = data.map(lambda doc: (int(doc[0]), doc[1].lower())).mapValues(schingling).mapValues(set)
        LSH.hash_alg = Minhashing(hash_function)
        LSH.band = bands
        LSH.rows = rows

        self.dataset = data_set
        self.rdd_dataset = self.sc.parallelize(self.dataset.collect())
        self.b_dataset = self.sc.broadcast(self.dataset.collect())
        self.row_list = self.get_rows(bands, rows)
        self.b_row_list = self.sc.broadcast(self.row_list)

        self.jaccard_similary_dict = {}

    @staticmethod
    def get_set_signatures(set, seed_list):
        _result = []
        _signatures = []
        _band = 0

        logging.debug('LSH.get_set_signatures=>seed_list=%s' % (seed_list))
        logging.debug('LSH.get_set_signatures=>set=%s' % (str(set)))
        for seed in seed_list:
            if _band < math.floor(seed/LSH.rows):
                _result.append((tuple(_signatures), set[0]))
                _signatures = [_band]
                _band += 1
            else: pass
            _signatures.append(LSH.hash_alg.get_value(set[1], seed)) # Get minhash signature for each row/seed
        _result.append((tuple(_signatures), set[0]))
        logging.debug('Minhash results=%s' % (_result))
        return _result

    def get_rows(self, bands, rows):
        # Return get_bands=[0, 1, 2, 3, 4, 5, 6, 7...N]
        _rows = []

        for i in range(bands):
            for j in range(rows):
                _rows.append(i*rows+j)

        logging.debug('LSH.get_rows=%s' % (_rows))
        return _rows


    def execute(self):
        _similar_sets_dict = {}
        _jaccard_similary_list = []
        _row_list = self.b_row_list.value
        _rdd_dataset = self.rdd_dataset
        _dataset = self.b_dataset.value
        logging.debug('LSH.execute=>_rdd_dataset =%s' % (str(_rdd_dataset)))

        _rdd_similar_set_candidate_list = _rdd_dataset.map(lambda x: LSH.get_set_signatures(x, _row_list)).flatMap(lambda x:
                                                                                                                   ((x[i][0], x[i][1]) for i in range(len(x)))).groupByKey().map(lambda x: tuple(x[1])).filter(lambda x: len(x)>1).distinct()
        logging.debug('LSH.execute=>_rdd_similar_set_candidate_list =%s' % (_rdd_similar_set_candidate_list.collect()))

        rdd_dataset = _rdd_similar_set_candidate_list.map(lambda candidate_sets: LSH.get_jaccard_similarity(_dataset, candidate_sets))
        _similar_sets_dict = rdd_dataset.flatMap(lambda x: x.items()).collectAsMap()
        logging.debug('LSH.execute=>_similar_sets_dict2=%s' % (_similar_sets_dict))
        return _similar_sets_dict


    @staticmethod
    def get_jaccard_similarity(dataset, candidates):
        # Input dataset to calculate similar sets based on candidates
        # create base set and its similar sets in a dictionary.
        # candidates= = (setA, setF, setG) =('2', '9', '10')
        # return = {base_set:(similar_set:jaccard_similarity, )}
        _similar_dict = {}
        _result = []
        _dataset_dict = dict(dataset)
        _set = None
        _similar_sets = None
        _total_set_list = []
        _counter = 0
        _jaccard_similarity = 0.0
        logging.debug('LSH.get_jaccard_similarity=>candidates=%s' % (str(candidates)))
        logging.debug('type(_dataset_dict)=%s, _dataset_dict = %s' % (type(_dataset_dict),_dataset_dict))

        # Generate combination for each set in candidate sets.
        for i in range(len(candidates)):
            _b_set = candidates[i] #base set
            _result = []
            for j in range(len(candidates)):
                if i != j:
                    _s_set = candidates[j] #similar set
                    _jaccard_similarity = 0.0
                    _total_set_list = []

                    #calculate jaccard similarity.
                    if tuple((_b_set, _s_set)) in LSH.jaccard_similarity:
                        _jaccard_similarity = LSH.jaccard_similarity[(_b_set, _s_set)]
                    else:   #calculate jaccard similarity
                        _b_items = _dataset_dict[_b_set]
                        _s_items = _dataset_dict[_s_set]
                        _total_set_list.append(set(_b_items))
                        _total_set_list.append(set(_s_items))
                        _jaccard_similarity = float(len(set.intersection(*_total_set_list))/len(set.union(*_total_set_list)))
                        # Put calculation results into local cache
                        LSH.jaccard_similarity.update({tuple((_b_set, _s_set)):_jaccard_similarity})
                        LSH.jaccard_similarity.update({tuple((_s_set, _b_set)):_jaccard_similarity})

                    logging.debug('jaccard similarity _result=%s' % (LSH.jaccard_similarity))
                else: pass

            _similar_dict[_b_set]=LSH.jaccard_similarity
        logging.debug('LSH.get_jaccard_similarity=> _similar_dict=%s' % (_similar_dict))
        return _similar_dict

if __name__ == '__main__':

    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename',
                         default=None)

    (options, args) = optparser.parse_args()

    if options.input is None:
        FILE_NAME_TO_READ = sys.stdin
    elif options.input is not None:
        FILE_NAME_TO_READ = os.path.join(DIRECTORY_TO_SAVE, options.input)

    if PRINT_TIME: logging.debug('Locality Sensitive Hashing =>Start=>%s' % (str(datetime.now())))
    logging.info('Locality Sensitive Hashing =>Start=>%s' % (str(datetime.now())))
    jaccard_similarity_dict = {}

    lsh = LSH(BANDS, ROWS)
    jaccard_similarity_dict = lsh.execute()
    setOutputData(OUTPUT_FILE, jaccard_similarity_dict)
    if PRINT_TIME: logging.debug('Locality Sensitive Hashing =>Finish=>%s' % (str(datetime.now())))
    logging.info('Locality Sensitive Hashing =>Finish=>%s' % (str(datetime.now())))
