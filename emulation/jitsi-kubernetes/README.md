# jitsi-kubernetes-scalable-service

## Kubernetes deployment for scalable videobridges

### Requirements
1. Azure account (Pay-as-you-go to be able to deploy 5 nodes) with azure-cli and kubectl installed and setup properly
2. Docker Environment


### Installation Guide Specific for Azure AKS
1. Build the patched Jitsi Kubernetes server images

    1. Build JVB image

        ``cd ../dockers/jitsi-dockers/jvb-docker``
        
        ``docker build . -t jvb:1.3.0``
        
        ``cd ../../../jitsi-kubernetes``

    1. Build Jibri image
    
        ``cd ../dockers/jitsi-dockers/jibri-docker``
        
        ``docker build . -t jibri:1.2.0``
        
        ``cd ../../../jitsi-kubernetes``

    1. Build Autoscaler image (Replace <autoscaler-algo> with either 'baseline' or 'dqn')
    
        ``cd ../dockers/autoscaler-algo-dockers/<autoscaler-algo>-docker``
        
        ``docker build . -t <autoscaler-algo>-autoscaler:1.0.0``
        
        ``cd ../../../jitsi-kubernetes``

2. Create Azure resources
 
    2. Create Azure Group

        `az group create --name JitsiGroup --location eastus`
  
    2. Create Azure Container Repository (ACR)

        `az acr create --resource-group JitsiGroup --name jitsiacr --sku Basic`
        
    2. Push patched images to ACR

        `az acr login --name jitsiacr`
        
        `docker tag jvb:1.3.0 jitsiacr.azurecr.io/jvb:1.3.0`
        
        `docker tag jibri:1.2.0 jitsiacr.azurecr.io/jibri:1.2.0`
        
        `docker tag <autoscaler-algo>-autoscaler:1.0.0 jitsiacr.azurecr.io/<autoscaler-algo>-autoscaler:1.0.0`
        
        `docker push jitsiacr.azurecr.io/jvb:1.3.0`
        
        `docker push jitsiacr.azurecr.io/jibri:1.2.0`
        
        `docker push jitsiacr.azurecr.io/<autoscaler-algo>-autoscaler:1.0.0`
        
    2. Create Azure Kubernetes Service (AKS)

        `az aks create --name JitsiCluster --resource-group JitsiGroup --node-count 5 --generate-ssh-keys --attach-acr jitsiacr`
        
    2. Connect to AKS

        `az aks get-credentials --name JitsiCluster --resource-group JitsiGroup`

3. Configure Kubernetes node

   3. Change Label

	`kubectl get nodes -o wide` to get the node name as well as its IP address

	`kubectl label nodes <node-name> asound=alsa1`

   3. Update Linux & Sound Cards

	Follow the steps [here](https://docs.microsoft.com/en-us/azure/aks/ssh#create-the-ssh-connection) to get the SSH access to the node. Then follow the steps below inside the node,

	`sudo apt update && sudo apt install -y linux-image-extra-virtual`

	`sudo vim /etc/modprobe.d/alsa-loopback.conf` and write `options snd-aloop enable=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 index=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29` inside

	`sudo vim /etc/modules` and write `snd-aloop` at the bottom of the file

	`sudo vim /etc/default/grub` and replace `GRUB_DEFAULT=0` with `GRUB_DEFAULT="1>2"`

	`sudo update-grub` then `sudo reboot`

4. Create jitsi namespace & secrets

   `kubectl apply -f jitsi-init.yaml`

5. Deploy Ingress NGINX Controller

   `kubectl apply -f nginx-deploy.yaml`

6. Deploy Ingress HAProxy Controller

   `kubectl apply -f haproxy.yaml`

Then go to Azure portal -> all resources -> select the Public IP address resource with name that has prefix "kubernetes-".

For the ingress-nginx public IP address:

Go to the configuration tab and set the dns name, e.g. jitsicluster to get jitsicluster.eastus.cloudapp.azure.com domain.

For the ingress-haproxy public IP address:

Go to the configuration tab and set the dns name, e.g. jitsigrafana to get jitsigrafana.eastus.cloudapp.azure.com domain.

Change the placeholders in **dev/nginx-ingress.yaml** and **dev/haproxy-ingress.yaml** with the domain name you just get.

7. Deploy JVB and Web Services

   `kubectl apply -f kube-statefulset-deployment/service.yaml`

Wait until all the services get internal and external IP addresses.
Then replace all the placeholders in **kube-statefulset-deployment/jvb-statefullset.yaml** file.
Also, replace the placeholder in **kube-statefulset-deployment/jibri/configMap.yaml** file with the web service internal IP.

8. Deploy InfluxDB

   `kubectl apply -f kube-statefulset-deployment/influxdb.yaml`

Change the address in the **kube-statefulset-deployment/telegraf-sec-conf.yaml** and **kube-statefulset-deployment/jicofo-telegraf-sec-conf.yaml** with the InfluxDB address.

9. Deploy Configurations

   `kubectl apply -f kube-statefulset-deployment/telegraf-sec-conf.yaml`

   `kubectl apply -f kube-statefulset-deployment/jicofo-telegraf-sec-conf.yaml`

10. Deploy jitsi web, jicofo, & prosody 

   `kubectl apply -f kube-statefulset-deployment/web-jicofo-prosody.yaml`

11. Deploy autoscaler (Make sure all necessary configurations have been set)

   `kubectl apply -f autoscaler/<autoscaler-algo>/deploy.yaml`

12. Install cert-manager

   `kubectl create namespace cert-manager`
   
   `kubectl apply -f cert-manager.yaml --validate=false`

Replace all the placeholders in **dev/letsencrypt-cluster-issuer.yaml** and **dev/letsencrypt-certificate.yaml** files.

Replace all the placeholders in **dev/grafana-cluster-issuer.yaml** and **dev/grafana-cert.yaml** files.

13. Install Lets Encrypt for Jitsi Meet & Grafana

   `kubectl apply -f dev/letsencrypt-cluster-issuer.yaml`
   
   `kubectl apply -f dev/letsencrypt-certificate.yaml`

   `kubectl apply -f dev/grafana-cluster-issuer.yaml`
   
   `kubectl apply -f dev/grafana-cert.yaml`

14. Deploy Prometheus Stack

   `kubectl create -f kube-prometheus/manifests/setup`

   Wait until the monitoring components are running, then

   `kubectl create -f kube-prometheus/manifests/`

15. Deploy NGINX Ingress (For Jitsi Meet) & HAProxy Ingress (For Grafana)

   `kubectl apply -f dev/nginx-ingress.yaml`

   `kubectl apply -f dev/grafana-ingress.yaml`

16. Modify the **kube-statefulset-deployment/hpa.yaml** file as you need and deploy

   `kubectl apply -f kube-statefulset-deployment/hpa.yaml`

17. Deploy Jibri

   `kubectl apply -f kube-statefulset-deployment/jibri/configMap.yaml`

   `kubectl apply -f kube-statefulset-deployment/jibri/dnsConfig.yaml`

   `kubectl apply -f kube-statefulset-deployment/jibri/finalizeConfig.yaml`

   `kubectl apply -f kube-statefulset-deployment/jibri/storage.yaml`

   `kubectl apply -f kube-statefulset-deployment/jibri/persistentVolumeClaim.yaml`

   `kubectl apply -f kube-statefulset-deployment/jibri/a1-statefulset.yaml`

To access Prometheus, run the command below

   `kubectl --namespace monitoring port-forward svc/prometheus-k8s 9090`

Then access via http://localhost:9090.

To access Grafana, run the command below

   `kubectl --namespace monitoring port-forward svc/grafana 3000`

Then access via http://localhost:3000 and use the default grafana user:passwd of `admin:admin`.

---

After executing  above sucessfully,
jitsi web can be accesed via your domain (e.g. jitsicluster.eastus.cloudapp.azure.com)

Grafana can also be accesed via your second domain (e.g. jitsigrafana.eastus.cloudapp.azure.com)

To have the dashboard, add InfluxDB as the datasource via Grafana WebApp.
1 InfluxDB datasource named as **InfluxDB** with the internal IP of InfluxDB as the address (port 8086) and **jitsi_stats** as the database
Another 1 InfluxDB datasource named as **InfluxDB2** with the internal IP of InfluxDB as the address (port 8086) and **jicofo_stats** as the database
Then create the dashboard template by importing the **dpnm-dashboard.json** file.

After you are sure with the deployment, you can move to production env (for secure HTTPS) by using **prod/** folder instead of **dev/** folder

To delete the cluster, run `az group delete --name JitsiGroup --yes --no-wait` and then delete all resources in the **All resources** page in the Azure portal.
