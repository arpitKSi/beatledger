# Deploy BeatLedger

Use a public GitHub repository with the synthetic fixtures and the Streamlit Community Cloud entry point `app.py` on the default branch. Python 3.12 is the tested runtime. Publish the source and synthetic sample only. [Streamlit deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## Server settings

In Community Cloud, set this root-level secret to enable hosted mode.

```toml
BEATLEDGER_DEPLOYMENT = "hosted"
```

Hosted uploads run the offline baseline. **Judge sample live** uses the bundled synthetic scripts when a remote provider is configured. The saved model run remains available when the live provider is down or unconfigured.

For an approved free-tier provider, add these settings directly in the app’s Secrets panel. Use the endpoint, model identifier, and response format supported by that provider.

```toml
BEATLEDGER_BASE_URL = "https://provider.example/v1"
BEATLEDGER_MODEL = "provider-model-id"
BEATLEDGER_API_KEY = "paste-the-key-into-Streamlit-Secrets"
BEATLEDGER_RESPONSE_FORMAT = "json"
BEATLEDGER_TIMEOUT_SECONDS = "15"
BEATLEDGER_MAX_OUTPUT_TOKENS = "500"
```

The app judges at most four synthetic cuts per live run and reuses the result for 15 minutes per app process. The provider URL, model and credential are server settings. [Streamlit secrets documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

## Verify a published deployment

Open the deployment from a signed-out browser. Confirm that the bundled sample shows three episodes, that the saved model run has nine judged cuts, and that a fresh synthetic upload can produce downloadable CSV and JSON. Try **Judge sample live** with and without an operational provider. Confirm that an approved silent cut requires footage confirmation and that a trim resets its status. Run the test and batch commands in the public README from a fresh checkout.
