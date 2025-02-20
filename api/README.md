# Planwise API

## Instructions for running the API using Docker (recommended)

1. Make sure you have docker installed on your machine

2. Run the following command to build the docker container and the spin up a database instance

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

This will default the container to run on port 8080 and the database on port 5432. You can change these ports in the `docker-compose.dev.yml` file.
The application will run on detached mode.

3. To stop the container run the following command

```bash
docker compose -f docker-compose.dev.yml down
```

## Instructions for running the API using fastapi dev

1. Create a virtual environment

```bash
cd api
python3 -m venv venv
```

2.Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the API

```bash
fastapi dev main.py
```
