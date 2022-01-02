import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Run experiments on distributed systems',
    Svg: require('../../static/img/undraw_server_re_twwj.svg').default,
    description: (
      <>
        Run experiments on HPC clusters like SLURM, or use cloud providers like AWS, Azure or GCP to run large scale benchmarks.
      </>
    ),
  },
  {
    title: 'Easy to use',
    Svg: require('../../static/img/undraw_programmer_re_owql.svg').default,
    description: (
      <>
        fseval has an easy to understand API and can easily be extended. Define your own dataset adapters, callbacks and storage providers.
      </>
    ),
  },
  {
    title: 'Reproducible experiments',
    Svg: require('../../static/img/undraw_personal_settings_re_i6w4.svg').default,
    description: (
      <>
        All information to replay an experiment is neatly stored in a config file. Others
        can easily reproduce your results.
      </>
    ),
  },
];

const FeatureListBottom = [
  {
    title: 'Send experiment results directly to a dashboard',
    Svg: require('../../static/img/undraw_dark_analytics_re_2kvy.svg').default,
    description: (
      <>
        fseval integrates with <a href="https://wandb.ai">Weights and Biases</a>, so you can enjoy all the powerful tooling built into the platform to help analyze your data.
      </>
    ),
  },
  {
    title: 'Export your data to any SQL database',
    Svg: require('../../static/img/undraw_metrics_re_6g90.svg').default,
    description: (
      <>
        Experiment metrics can be sent to one of many SQL databases. Support includes SQLite, Postgresql, MySQL, Oracle and more. Support is achieved through integration with <a href="https://www.sqlalchemy.org/">SQL ALchemy</a>.
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <>
      <div className="text--center">
        <Svg className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </>
  );
}

export default function HomepageFeatures() {
  const bottomLeftFeature = FeatureListBottom[0];
  const bottomRightFeature = FeatureListBottom[1];

  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <div className={clsx('col col--4')}>
              <Feature key={idx} {...props} />
            </div>
          ))}
        </div>
        <div className="row">
          <div className={clsx('col col--4 col--offset-2')}>
            <Feature key={0} {...bottomLeftFeature} />
          </div>
          <div className={clsx('col col--4')}>
            <Feature key={0} {...bottomRightFeature} />
          </div>
        </div>
      </div>
    </section>
  );
}
