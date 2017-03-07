node {
        try {
           stage 'Checkout'
           checkout scm
           stage 'Tests'
           sh '''
           source /ihme/costeffectiveness/conda_env/bin/activate /ihme/costeffectiveness/conda_env
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH://ihme/costeffectiveness/conda_env/lib
           tox --recreate -e py35
           '''

           currentBuild.result = 'SUCCESS'
        } catch (err) {
        currentBuild.result = "FAILURE"
        sh '''

            /usr/sbin/sendmail "$(git --no-pager show -s --format='%an <%ae>' HEAD)" <<EOF
    Subject: Build failure on branch: ${BRANCH_NAME}

    Your commit ($(git rev-parse HEAD)) seems to have broken the build. Take a look: ${BUILD_URL}/
    EOF
        '''
        }
}
