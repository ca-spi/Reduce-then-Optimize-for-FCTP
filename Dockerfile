FROM gurobi/python:10.0.1_3.10

WORKDIR /home

RUN apt-get -y update && \
    apt-get install -y wget && \
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.5-linux-x86_64.tar.gz && \
    tar zxvf julia-1.8.5-linux-x86_64.tar.gz

ENV PATH="$PATH:/home/julia-1.8.5/bin"

ADD requirements.txt /home/requirements.txt
RUN pip install -r requirements.txt && \
    python -c "import julia; julia.install()"

RUN julia -e 'using Pkg; Pkg.add("Graphs"); Pkg.add("JuMP"); Pkg.add("Gurobi")'