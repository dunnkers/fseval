# ALL_RANKERS = ["chi2", "relieff", "tabnet"]

# @pytest.fixture(params=ALL_RANKERS)
# def ranker_config(config_loader, request):
#     ranker_config = get_config(config_loader, "ranker", request.param)
#     return ranker_config


# @pytest.fixture
# def classification_ranker(ranker_config):
#     ranker_config["task"] = Task.classification
#     ranker_object = instantiate(ranker_config)
#     return ranker_object
