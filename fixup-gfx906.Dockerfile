FROM mx-llama-rocm10-gfx906:local
RUN echo "/app" > /etc/ld.so.conf.d/llama-app.conf && ldconfig
ENV PATH="/app:/opt/rocm/bin:$PATH" \
    LD_LIBRARY_PATH="/app:/opt/rocm/lib"
