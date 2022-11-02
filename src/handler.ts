const handler = async function (event: any, context: any) {
    try {
        var method = event.httpMethod

        if (method === "GET") {
            if (event.path === "/") {

                return {
                    statusCode: 200,
                    headers: {},
                    body: JSON.stringify({ "y": "o" }),
                }
            }
        }

        return {
            statusCode: 400,
            headers: {},
            body: "We only accept GET /",
        }
    } catch (error) {
        let body
        if (error instanceof Error) {
            body = error.stack
        } else {
            body = JSON.stringify(error, null, 2)
        }
        return {
            statusCode: 400,
            headers: {},
            body: JSON.stringify(body),
        }
    }
}

export { handler }