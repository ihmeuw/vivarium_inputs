node {
        try {
           stage 'Checkout'
           checkout scm
           stage 'Tests'
           sh '''
           source /ihme/scratch/users/svcceci/.conda/bin/activate /ihme/scratch/users/svcceci/.conda/
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH://ihme/scratch/users/svcceci/.conda/lib/
           tox --recreate -e py36
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
