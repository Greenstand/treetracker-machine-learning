#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import xmltodict, os, sys, sqlite3, random
import tensorflow as tf

################################ GLOBALS ######################################

DATA_PATH = '/data/workspace/datasets/PlantCLEF_2016'
DB_PATH = os.path.join(DATA_PATH, 'PlantCLEF_2016.sqlite')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
	'class_target',
	'species',
	'The classification target for PlantCLEF2016, one of "species" (default), "family", "genus".')

tf.app.flags.DEFINE_string(
	'split_strategy',
	'InS',
	'The strategy for split creation (if class_target != "species"), one of "InS" (default), "ExS".')

tf.app.flags.DEFINE_string(
	'k_min',
	'2',
	'The #species threshhold for "ExS" strategy, integer >= "2", defaults to "2".')

###############################################################################

class xml(object):
    
    def read(xml_file):
        xml_data = open(xml_file, 'r', encoding='utf-8')
        return xml.parse(xml_data.read())
    
    def parse(xml_data):
        parsed = xmltodict.parse(xml_data)
        return parsed

###############################################################################

def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path

###############################################################################

class db:
    
    def __init__(self, filename='PlantCLEF2016.sqlite'):
        self.filename = filename
        self._connect()
        self.execute('PRAGMA busy_timeout = 100')
    
    def _connect(self):
        self.connection = sqlite3.connect(self.filename, check_same_thread=False)
        self.cursor = self.connection.cursor()
        self._connected = True
        self.create_tables()
    
    def _close(self):
        if self._connected:
            self.connection.commit()
            self.cursor.close()
            self.connection.close()
            self._connected = False
    
    def execute(self, sql, values=None):
        if values:
            self.cursor.execute(sql, values)
        else:
            self.cursor.execute(sql)
        self.connection.commit()
        
    def select(self, sql, values=None):
        if values:
            self.cursor.execute(sql, values)
        else:
            self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        return rows
        
    def create_tables(self):
        self.execute("pragma foreign_keys=on")        
        self.execute("""
                  CREATE TABLE IF NOT EXISTS images
                  (
                        id INTEGER PRIMARY KEY,
                        filepath VARCHAR,
                        family_id INTEGER,
                        genus_id INTEGER,
                        species_id INTEGER,
                        observation_id INTEGER, 
                        content_id INTEGER,
                        vote INTEGER, 
                        orig_learntag_id INTEGER,
                        FOREIGN KEY(family_id) REFERENCES families(id),
                        FOREIGN KEY(genus_id) REFERENCES genera(id),
                        FOREIGN KEY(species_id) REFERENCES species(id),
                        FOREIGN KEY(content_id) REFERENCES contents(id),
                        FOREIGN KEY(orig_learntag_id) REFERENCES orig_learntags(id)
                   )
                  """)
        for table in ['families', 'genera', 'species', 'contents', 'orig_learntags']:
            self.execute("""
                         CREATE TABLE IF NOT EXISTS {}
                         (
                             id INTEGER PRIMARY KEY,
                             name VARCHAR UNIQUE
                         )
                         """.format(table))
        
    def insert_table_val(self, table,column,data):
        self.execute(
                'INSERT OR IGNORE INTO "{}" ("{}") VALUES (?);'.format(table, column),
                data)
        row_id = self.select(
                'SELECT id FROM "{}" WHERE "{}" = ?'.format(table, column),
                data)
        if row_id:
            row_id = row_id[0]
        return row_id

    def insert_species(self,data):
        self.execute(
                "INSERT OR IGNORE INTO species (id, name) VALUES (?,?);",
                data)
    
    def insert_image(self,data):
        fam_id = self.insert_table_val('families', 'name', (data[2],))[0]
        gen_id = self.insert_table_val('genera', 'name', (data[3],))[0]
        self.insert_species((int(data[7]),data[4],))
        content_id = self.insert_table_val('contents', 'name', (data[5],))[0]
        olt_id = self.insert_table_val('orig_learntags', 'name', (data[6],))[0]
        self.execute(
                "INSERT OR IGNORE INTO images "
                "VALUES ("
                "?,"#id
                "?,"#filepath
                "?,"#family_id
                "?,"#genus_id
                "?,"#species_id
                "?,"#observation_id
                "?,"#content_id
                "?,"#vote
                "?)"#orig_learntag_id
                , (data[0], 
                   data[1], 
                   fam_id, 
                   gen_id, 
                   int(data[7]), 
                   data[8],
                   content_id, 
                   data[9],
                   olt_id
                   )
                )

###############################################################################

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '%'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

###############################################################################

def analyze_dir(basedir, my_db):
    for root, dirs, files in os.walk(basedir):
        print('scanning "{}"'.format(root))
        files = [file for file in files if '.xml' in file and file[0]!='.']
        num_items = len(files)
        print(' {} folders, {} files'.format(len(dirs), num_items))
        for num, file in enumerate(files):
            file = os.path.join(root, file)
            img_file = file.replace('.xml', '.jpg')
            if not os.path.exists(img_file):
                print("WARNING: File does not exist: ", img_file)
                continue
            xml_data = xml.read(file)
            xml_data = xml_data["Image"]
            if xml_data["ClassId"] != '9999' and all(xml_data[x] for x in ['Family', 'Genus']):
                my_db.insert_image( 
                        (
                            xml_data["MediaId"],
                            img_file.replace(basedir+'/', ''),
                            xml_data["Family"],
                            xml_data["Genus"],
                            xml_data["Species"],
                            xml_data["Content"],
                            xml_data["LearnTag"],
                            xml_data["ClassId"],
                            xml_data["ObservationId"],
                            xml_data["Vote"]
                        )
                        )
            printProgressBar(num, num_items, prefix = 'Progress:', suffix = 'Complete', length = 50)

###############################################################################

def run(dataset_dir):
    """Creates directory tree filled with symlinks that can be parsed by tf dataset_convert.DatasetReader.
    
    Args:
        dataset_dir: The dataset directory where the dataset shall be stored.
    """
    if not FLAGS.class_target:
        class_target = 'species'
    else:
        class_target = FLAGS.class_target
    print('classification target = ', class_target)

    if not FLAGS.split_strategy or class_target == 'species':
        split_strategy = 'InS'
    else:
        split_strategy = FLAGS.split_strategy
    print('split creation strategy = ', split_strategy)

    my_db = db(filename=DB_PATH)

    if class_target == 'species':
        class_table = 'species'
    elif class_target == 'family':
        class_table = 'families'
    elif class_target == 'genus':
        class_table = 'genera'
    num_classes = my_db.select("SELECT COUNT(DISTINCT id) FROM {}".format(class_table))[0][0]
    class_id_formatstr = "0%id" % len(str(num_classes))

    dataset_dir = os.path.join(dataset_dir, class_target, split_strategy)
    mkdir(dataset_dir)
    dataset_dir = os.path.join(dataset_dir, 'grouped_%03d' % sum(['grouped' in x for x in os.listdir(dataset_dir)]))
    print('creating new directory structure at ', dataset_dir)
    mkdir(dataset_dir)

    # save class idx vs labels to file
    fid = open(os.path.join(dataset_dir, 'class_idx_vs_labels.txt'), 'w')

    if split_strategy == 'ExS':
        [mkdir(os.path.join(dataset_dir, subdir)) for subdir in ['train', 'validation']]
        if not FLAGS.k_min:
            k_min = 2
        else:
            k_min = FLAGS.k_min
        print('species threshhold k_min = ', k_min)

        query_images = """
            WITH target_ids AS (
                SELECT 
                    {0}_id AS tid, 
                    COUNT(DISTINCT(species_id)) AS counter 
                FROM images 
                GROUP BY {0}_id 
                HAVING counter >= {1})
            SELECT 
                filepath, species_id, {0}_id
            FROM images
            WHERE {0}_id IN (SELECT tid FROM target_ids)
            """.format(class_target, k_min)
        all_images = my_db.select(query_images)
        class_ids = list(set(x[2] for x in all_images))
        for class_idx,class_id in enumerate(class_ids):
            fid.write("%i;%i\n" % (class_idx, class_id))
            # get the species_ids
            sp_list = list(set(x[1] for x in all_images if x[2]==class_id))
            # test set = random 10% of species
            numel = int(max(round(0.1 * len(sp_list)), 1))
            sp_sel = random.sample(sp_list, numel)
            img_dir_test = mkdir(os.path.join(dataset_dir, 'validation', format(class_idx, class_id_formatstr)))
            for sp in sp_sel:
                sp_list.remove(sp)
                imgs = [x for x in all_images if x[1]==sp]
                for img in imgs:
                    os.symlink(os.path.join(DATA_PATH, img[0]), os.path.join(img_dir_test, os.path.basename(img[0])))

            # train set = remaining 90%
            img_dir = mkdir(os.path.join(dataset_dir, 'train', format(class_idx, class_id_formatstr)))
            for sp in sp_list:
                imgs = [x for x in all_images if x[1]==sp]
                for img in imgs:
                    os.symlink(os.path.join(DATA_PATH, img[0]), os.path.join(img_dir, os.path.basename(img[0])))
            printProgressBar(class_idx, len(class_ids), prefix = 'Progress:', suffix = 'Complete', length = 50)

    elif split_strategy == 'InS':
        [mkdir(os.path.join(dataset_dir, subdir)) for subdir in ['train', 'validation']]
        query_images = "SELECT filepath, {}_id FROM images".format(class_target)
        all_images = my_db.select(query_images)
        class_ids = list(set(x[1] for x in all_images))
        for class_idx,class_id in enumerate(class_ids):
            fid.write("%i;%i\n" % (class_idx, class_id))
            imgs = [x for x in all_images if x[1]==class_id]
            # test set = random 10% of images
            numel = int(max(round(0.1 * len(imgs)), 1))
            img_sel = random.sample(imgs, numel)
            img_dir = mkdir(os.path.join(dataset_dir, 'validation', format(class_idx, class_id_formatstr)))
            for img in img_sel:
                imgs.remove(img)
                os.symlink(os.path.join(DATA_PATH, img[0]), os.path.join(img_dir, os.path.basename(img[0])))
            
            # train set = remaining 90%
            img_dir = mkdir(os.path.join(dataset_dir, 'train', format(class_idx, class_id_formatstr)))
            for img in imgs:
                os.symlink(os.path.join(DATA_PATH, img[0]), os.path.join(img_dir, os.path.basename(img[0])))
            printProgressBar(class_idx, len(class_ids), prefix = 'Progress:', suffix = 'Complete', length = 50)

    fid.close()
    my_db._close()
    print('DONE. Symlinks for {} classes created at "{}"'.format(class_idx + 1 , dataset_dir))
    return dataset_dir

###############################################################################

if __name__ == '__main__':
    
    if not os.path.isfile(DB_PATH):
        # create new database by parsing the XMLs in DATA_PATH
        my_db = db(filename=DB_PATH)
        analyze_dir(DATA_PATH, my_db)
        my_db.execute('DELETE FROM genera WHERE name IS NULL')
        my_db.execute('DELETE FROM families WHERE name IS NULL')
        my_db.execute('DELETE FROM species WHERE name IS NULL')
        my_db.execute('DELETE FROM images WHERE family_id IS NULL')
        my_db.execute('VACUUM')
        my_db._close()
        
    run(sys.argv[1])