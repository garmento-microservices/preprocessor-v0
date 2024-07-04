ARG PYTHON_BASE=3.10-slim
# Base Image (with base APT packages)
FROM python:$PYTHON_BASE AS base

# install required dependencies
RUN apt update && apt upgrade -y
RUN apt install git gcc g++ python3-dev default-libmysqlclient-dev pkg-config \
    ffmpeg libsm6 libxext6 wget -y

# build stage
FROM base AS builder

# install PDM
RUN pip install -U pdm==2.15.0
# disable update check
ENV PDM_CHECK_UPDATE=false
# copy files
COPY pyproject.toml README.md /project/

# install dependencies and project into the local packages directory
WORKDIR /project
RUN pdm install --prod --no-editable --no-self --no-isolation
COPY src/ /project/src

# run stage
FROM base

# retrieve packages from build stage
COPY --from=builder /project/.venv/ /project/.venv
ENV PATH="/project/.venv/bin:$PATH"
# set command/entrypoint, adapt to fit your needs
COPY src /project/src
COPY .env /project/
COPY ./configs/ /project/configs/
COPY ./models/ /project/models/
COPY ./posenet_models/ /project/posenet_models/
COPY tmp/weights.zip /root/.cache/torch/hub/checkpoints/weights.zip
ADD ./entry.sh ./alembic.ini /project/
RUN chmod +x /project/entry.sh
WORKDIR /project
COPY ./presets ./presets
CMD [ "./entry.sh" ]
