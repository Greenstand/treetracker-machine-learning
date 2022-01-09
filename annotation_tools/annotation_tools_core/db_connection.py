import psycopg2
import sshtunnel
from config import config

def create_ssh_tunnel(credentials_file):

    params = config(credentials_file, 'sshtunnel')

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


def connect(credentials_file):
    """ Connect to the PostgreSQL database server """
    conn = None
    cur  = None
    try:
        # read connection parameters
        params = config(credentials_file, 'postgresql')

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


def execute_query(cur, query):

    cur.execute(query)

    res = []
    query_result = cur.fetchone()

    while query_result is not None:
        res.append(query_result)
        query_result = cur.fetchone()

    return res
