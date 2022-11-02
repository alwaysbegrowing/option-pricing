// import * as cdk from 'aws-cdk-lib';
// import { Construct } from 'constructs';
// // import * as sqs from 'aws-cdk-lib/aws-sqs';

// export class CalcStack extends cdk.Stack {
//   constructor(scope: Construct, id: string, props?: cdk.StackProps) {
//     super(scope, id, props);

//     // The code that defines your stack goes here

//     // example resource
//     // const queue = new sqs.Queue(this, 'CalcQueue', {
//     //   visibilityTimeout: cdk.Duration.seconds(300)
//     // });
//   }
// }

import { LambdaIntegration, MethodLoggingLevel, RestApi } from "aws-cdk-lib/aws-apigateway"
import { Function, Runtime, AssetCode, } from "aws-cdk-lib/aws-lambda"
import { Duration, Stack, StackProps } from "aws-cdk-lib"
import { Construct } from "constructs"

interface LambdaApiStackProps extends StackProps {
  functionName: string
}

export class CalcStack extends Stack {
  private restApi: RestApi
  private lambdaFunction: Function

  constructor(scope: Construct, id: string, props: LambdaApiStackProps) {
    super(scope, id, props)


    this.restApi = new RestApi(this, this.stackName + "RestApi", {
      deployOptions: {
        stageName: "beta",
        metricsEnabled: true,
        loggingLevel: MethodLoggingLevel.INFO,
        dataTraceEnabled: true,
      },
    })


    this.lambdaFunction = new Function(this, props.functionName, {
      functionName: props.functionName,
      handler: "handler.handler",
      runtime: Runtime.NODEJS_16_X,
      code: new AssetCode(`./src`),
      memorySize: 512,
      timeout: Duration.seconds(10),
    })

    this.restApi.root.addMethod("GET", new LambdaIntegration(this.lambdaFunction, {}))
  }
}