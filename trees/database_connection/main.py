#!/usr/bin/python
import psycopg2
import sshtunnel
from config import config
import os

"""
This script requires you to have a separate file called "database.ini" with credentials for connecting.

[postgresql]
host=localhost
database=<database>
user=<user>
port=<local_port>
password=<password>

[sshtunnel]
ssh_address_or_host=<tunnel_host>
ssh_username=<user for tunnel connection>
ssh_password=<remote_gateway_password>
ssh_port=<tunnel_port>
ssh_pkey=<path_to_private_key>
remote_bind_address=<remote_server>
remote_bind_address_port=<remote_port>
local_bind_address=localhost
local_bind_address_port=<local_port>
"""

def get_query(last_img_time):

    return "SELECT uuid, time_created, date_part('epoch', time_created), image_url \
            FROM trees  \
            WHERE planter_id IN (select id from planter where organization_id = 194) AND date_part('epoch', time_created) > " \
            + str(last_img_time) \
            + "ORDER BY date_part('epoch', time_created) ASC;"


def create_ssh_tunnel():

    params = config('database.ini', 'sshtunnel')

    new_tunnel = sshtunnel.SSHTunnelForwarder(
        ssh_address_or_host=params['ssh_address_or_host'],
        ssh_config_file=None,
        ssh_username=params['ssh_username'],
        ssh_password=params['ssh_password'],
        ssh_port=params['ssh_port'],
        ssh_pkey=params['ssh_pkey'],
        remote_bind_address=(params['remote_bind_address'],int(params['remote_bind_address_port'])),
        local_bind_address = (params['local_bind_address'], int(params['local_bind_address_port'])),
    )

    return new_tunnel


def connect_to_db():
    """ Connect to the PostgreSQL database server """
    conn = None
    cur  = None
    try:
        # read connection parameters
        params = config('database.ini', 'postgresql')

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    return (conn, cur)

def close_connection(conn, cur):
    if cur is not None:
        # close the communication with the PostgreSQL
        cur.close()

    if conn is not None:
        conn.close()
        print('Database connection closed.')


def execute_query(cur, last_timestamp):

    query = get_query(last_timestamp)
    cur.execute(query)

    res = []
    query_result = cur.fetchone()

    while query_result is not None:
        res.append(query_result)
        query_result = cur.fetchone()

    return res

def update_query_log(file_name,results):
    new_last_timestamp = int(results[-1][2])

    with open(file_name, "a") as f:
        f.write(str(new_last_timestamp) + "\n")

def create_new_cvat_task_list(results):

    img_urls = []

    for tree in results:
        img_urls.append(tree[3])

    return img_urls

if __name__ == '__main__':

    file_name = "log_queries.txt"

    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            this_timestamp = 0

            # read all logged timestamps
            lines = f.readlines()
            # extract last timestamp used to
            # create a task and remove '\n'
            last_timestamp = lines[-1][:-1]

    else:
        last_timestamp = "0"
        with open(file_name, "a") as f:
            f.write(last_timestamp + "\n")

    tunnel = create_ssh_tunnel()
    tunnel.start()
    (connection, cur) = connect_to_db()
    results = execute_query(cur, last_timestamp)

    if len(results) > 0:
        imgs_urls = create_new_cvat_task_list(results)
        update_query_log(file_name, results)

        with open("img_urls.txt", "a") as f:
            for this_url in imgs_urls:
                f.write(this_url + "\n")

    close_connection(connection, cur)
    tunnel.close()
