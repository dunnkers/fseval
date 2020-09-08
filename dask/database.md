
## Installing MongoDB chart
Using helm:

```shell
helm install my-mongodb --set architecture=standalone,useStatefulSet=true,externalAccess.enabled=true bitnami/mongodb
```

Now open up a `LoadBalancer` with external IP to the stateful set.
Locally, set `MONGODB_ROOT_PASSWORD` and `MONGODB_IP` environment variables in `bashrc` or `zshrc`.