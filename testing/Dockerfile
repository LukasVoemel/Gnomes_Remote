# Install python
FROM python:3.9

# Install dependencies (outside of submission speeds up process)
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy parent directory
ADD . /dragg-comp-submission

# Install submission
RUN pip install /dragg-comp-submission/.

# Run Tests
CMD python /dragg-comp-submission/testing/test_submission.py
