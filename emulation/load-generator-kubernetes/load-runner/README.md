## To create repo
`aws ecr create-repository --repository-name <repo-and/or-image-name> --image-scanning-configuration scanOnPush=true --region <region-id> --profile <aws-profile>`

## To login to repo
`aws ecr get-login-password --region <region-id> --profile <aws-profile> | docker login --username AWS --password-stdin <acc-id>.dkr.ecr.us-east-1.amazonaws.com`

## Build images
``cd ../../dockers/load-generator-dockers/torture-selenium-docker``

``docker build . -t torture-selenium:1.0.0``

``cd ../load-runner-docker``

``docker build . -t load-runner:1.0.0``

``cd ../../load-generator-kubernetes/load-runner``


## To create cluster
`eksctl create cluster --config-file=aws_eks_cluster.yaml --profile=<aws-profile>`

## To deploy torure-selenium dockers
`kubectl apply -f rbac.yaml`
`kubectl apply -f deploy.yaml`
