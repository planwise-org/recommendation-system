# Planwise API

## Instructions for running the API locally using Docker (highly recommended)

1. Make sure you have docker installed on your machine

2. Cd into api directory: `cd api`

3. Set up supabase locally

- Install supabase

MacOS with homebrew: `brew install supabase/tap/supabase`
Windows with scoop:
```bash
scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
scoop install supabase
```

- Initialize supabase: `supabase init`
- Start supabase: `supabase start`
- Copy the DB URL from supabase cli and paste it in the `DATABASE_URL` variable in the `docker-compose.dev.yml` file. **Remove 127.0.0.1 and replace with host.docker.internal**

4. Run the following command to build the docker container and the spin up a database instance

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

This will default the container to run on port 8080 and the database on port 5432. You can change these ports in the `docker-compose.dev.yml` file.
The application will run on detached mode.

5. To stop the container run the following command

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
