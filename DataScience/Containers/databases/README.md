# Databases


## Mongo DB
MongoDB is a cross-platform document-oriented database program. Classified as a NoSQL database program, MongoDB uses JSON-like documents with optional schemas. MongoDB is developed by MongoDB Inc.

### How to use MongoDB on Polaris
1. 


### Mongoku
MongoDB client for the web. Query your data directly from your browser. You can host it locally, or anywhere else, for you and your team.

It scales with your data (at Hugging Face we use it on a 1TB+ cluster) and is blazing fast for all operations, including sort/skip/limit. Built on TypeScript/Node.js/Angular.

#### How to use Mongoku on Polaris

1. To run mongoku directly, use the following command. Please edit the mongodb default host settings:

```bash
singularity run --overlay overlay.img -B $PWD/data:/data --env MONGOKU_DEFAULT_HOST="mongodb://user:password@myhost.com:8888" --env MONGOKU_SERVER_PORT=8000 mongoku.sif 
```

2. To view mongoku database on your browser and interact with the database you can ssh tunnel to the login node where the service is running. It can be polaris-login-XX as shown belowRun the container directly

```bash
export PORT=8000; ssh -L "localhost:${PORT}:localhost:${PORT}" atanikanti@polaris-login-02.alcf.anl.gov
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
