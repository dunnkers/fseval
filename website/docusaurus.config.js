// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'fseval',
  tagline: 'Benchmarking framework for Feature Selection and Feature Ranking algorithms 🚀',
  url: 'https://dunnkers.com',
  baseUrl: '/fseval/',
  trailingSlash: true,
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'dunnkers', // Usually your GitHub org/user name.
  projectName: 'fseval', // Usually your repo name.

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          sidebarCollapsed: true,
          // Please change this to your repo.
          editUrl: 'https://github.com/dunnkers/fseval/tree/website/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        gtag: {
          trackingID: '40680192-2',
          anonymizeIP: true,
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'fseval',
        logo: {
          alt: 'fseval Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'quick-start',
            position: 'left',
            label: 'Docs',
          },
          // {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://github.com/dunnkers/fseval',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Quick start',
                to: '/docs/quick-start',
              },
              {
                label: 'API reference',
                to: '/docs/main',
              },
              // {
              //   label: 'Recipes',
              //   to: '/docs/recipes/algorithm-stability',
              // },
            ],
          },
          {
            title: ' ',
            items: [
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/dunnkers/fseval',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} fseval`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};

module.exports = config;
