# TrainKernel development

- Experimental compiler research. Report measured results and limitations accurately.
- Preserve user changes and Claude compatibility files.
- Never commit credentials, private keys, .env files, or provider tokens.
- Run focused CPU compiler tests and GPU parity checks after relevant changes.
- Benchmark complete forward/backward/update steps against compiled PyTorch with CUDA Graphs where supported.
- Keep GPU allocation bounded and release project-created resources after results are saved.
- Avoid em dashes in project prose.
