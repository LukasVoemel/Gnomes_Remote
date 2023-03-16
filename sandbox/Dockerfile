# Install python
FROM python:3.9

# Install dependencies (outside of submission speeds up process)
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy parent directory
ADD . /dragg-comp-submission

# Install submission
RUN pip install /dragg-comp-submission/.

# Run simulation
WORKDIR  /dragg-comp-submission/sandbox/simulation
CMD python run_aggregator.py -r "redis://redis" & python run_player.py -r "redis://redis" 
