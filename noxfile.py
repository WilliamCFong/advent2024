import nox


@nox.session
def tests(session):
    session.install("pytest", ".")
    session.run("pytest", "-n", "auto")


@nox.session
def lint(session):
    session.install("flake8")
    session.run("flake8", "src", "tests")
