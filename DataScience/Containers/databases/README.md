# Databases
We provide singularity containers and steps to interact with database containers on Polaris

## Mongo DB
MongoDB is a cross-platform document-oriented database program. Classified as a NoSQL database program, MongoDB uses JSON-like documents with optional schemas. MongoDB is developed by MongoDB Inc.

### How to use MongoDB on Polaris
1. Copy the container to your home or project directory 
```bash
cp /soft/containers/mongo.sif /grand/<project_name>
```

2. Create a data and logs directory to bind to the running container
```bash
mkdir -p /grand/<project_name>/data
```

3. Running mongo
```bash
singularity exec --bind $PWD/data:/data/db mongo.sif mongod
```

To run it as an instance in the background. You can
```bash
singularity instance start --bind $PWD/data:/data/db mongo.sif mongoinstance
```

To stop the instance
```bash
singularity instance stop mongoinstance
```

To list all running instances
```bash
singularity instance list
```


### PyMongo
Pymongo is the Python client/library to connect to a running mongodb. Query and interact with your data directly from Python. Here's an example

```bash
> module load conda
> python3 -m venv ~/envs/mongoenv
> source ~/envs/mongoenv/bin/activate
> python3
>>> import pymongo 
>>> import pprint
>>> mongo_uri = "mongodb://localhost:27017/" 
>>> dbclient = pymongo.MongoClient(mongo_uri)
>>> appdb = dbclient["blog"]
>>> appcoll = appdb["blogcollection"]
>>> document = {"user_id": 1, "user": "test"}
>>> appcoll.insert_one(document)
<pymongo.results.InsertOneResult object at 0x7f3b63469790>
>>> dbclient.list_database_names()
['admin', 'blog', 'config', 'local']
```


## Neo4j
Neo4j is the world's leading open source Graph Database which is developed using Java technology. It is highly scalable and schema free (NoSQL).

### What is a Graph Database?
A graph is a pictorial representation of a set of objects where some pairs of objects are connected by links. It is composed of two elements - nodes (vertices) and relationships (edges).

Graph database is a database used to model the data in the form of graph. In here, the nodes of a graph depict the entities while the relationships depict the association of these nodes.

### How to use Neo4j on Polaris
1. Copy the container to your home or project directory 
```bash
cp /soft/containers/neo4j.sif /grand/<project_name>
```

2. Create a data and logs directory to bind to the running container
```bash
mkdir -p /grand/<project_name>/data
mkdir -p /grand/<project_name>/logs
```

3. Create a persistent overlay for the container. A persistent overlay is a directory or file system image that “sits on top” of your immutable SIF container. When you install new software or create and modify files the overlay will store the changes.
```bash
singularity overlay create --size 1024 overlay.img
```

4. Run the container
```bash
singularity exec -B $PWD/data:/data -B $PWD/logs:/logs --overlay overlay.img neo4j.sif neo4j start
Directories in use:
home:         /var/lib/neo4j
config:       /var/lib/neo4j/conf
logs:         /var/lib/neo4j/logs
plugins:      /var/lib/neo4j/plugins
import:       /var/lib/neo4j/import
data:         /var/lib/neo4j/data
certificates: /var/lib/neo4j/certificates
licenses:     /var/lib/neo4j/licenses
run:          /var/lib/neo4j/run
Starting Neo4j.
WARNING: Max 16384 open files allowed, minimum of 40000 recommended. See the Neo4j manual.
Started neo4j (pid:50867). It is available at http://localhost:7475
There may be a short delay until the server is ready.
```
5. To view neo4j database on your browser and interact with the database you can ssh tunnel to the login node where the service is running. It can be polaris-login-XX as shown below

```bash
export PORT=7475 export PORT1=7687; ssh -L "localhost:${PORT}:localhost:${PORT}" -L "localhost:${PORT1}:localhost:${PORT1}" <username>@<polaris-login-01>.alcf.anl.gov
```
