import { LambdaIntegration, RestApi, Cors } from "aws-cdk-lib/aws-apigateway";
import { Function, Runtime, AssetCode } from "aws-cdk-lib/aws-lambda";
import { PythonFunction } from "@aws-cdk/aws-lambda-python-alpha";
import { Duration, Stack, StackProps } from "aws-cdk-lib";
import { Construct } from "constructs";
import { ApiGateway } from "aws-cdk-lib/aws-events-targets";

interface LambdaApiStackProps extends StackProps {
  functionName: string;
}

export class CalcStack extends Stack {
  private restApi: RestApi;
  private lambdaFunction: Function;

  constructor(scope: Construct, id: string, props: LambdaApiStackProps) {
    super(scope, id, props);

    this.restApi = new RestApi(this, this.stackName + "RestApi", {
      cloudWatchRole: true,

      defaultCorsPreflightOptions: {
        allowOrigins: Cors.ALL_ORIGINS,
        allowMethods: Cors.ALL_METHODS, // this is also the default
      },

      // deployOptions: {
      //   stageName: "beta",
      //   metricsEnabled: true,
      //   loggingLevel: MethodLoggingLevel.INFO,
      //   dataTraceEnabled: true,
      // },
    });

    const calculatorBonds = new PythonFunction(this, "calculator", {
      functionName: props.functionName,
      handler: "handler",
      index: "handler.py",
      runtime: Runtime.PYTHON_3_9,
      entry: "./src",
      memorySize: 512,
      timeout: Duration.seconds(10),
    });

    this.restApi.root.addMethod(
      "GET",
      new LambdaIntegration(calculatorBonds, {
        // requestParameters: {
        //   "integration.request.path.v": "method.request.path.v",
        //   "integration.request.path.t": "method.request.path.t",
        //   "integration.request.path.fs": "method.request.path.fs",
        // },
      })
    );
  }
}
