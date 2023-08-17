import pytest

def pytest_addoption(parser):
    parser.addoption( "--dir",action="store")
    parser.addoption( "--num",action="store")
    parser.addoption( "--days",action="store")

@pytest.fixture()
def dir(request):
    return request.config.getoption("--dir")

@pytest.fixture()
def num(request):
    return request.config.getoption("--num")

@pytest.fixture()
def days(request):
    return request.config.getoption("--days")