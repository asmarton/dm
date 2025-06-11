import { createChart, LineSeries } from "lightweight-charts";


/**
 * @param {HTMLElement} containerEl
 * @param {{index: any[], columns: string[], data: any[]}} portfolio
 */
function setupChart(containerEl, portfolio) {
  const chart = createChart(containerEl);
  const seriesData = {};
  let balanceColumns = [];
  for (let i = 0; i < portfolio.columns.length; i++) {
    const column = portfolio.columns[i];
    if (column.endsWith('Balance')) {
      balanceColumns.push({column, index: i});
    }
  }

  for (const { column } of balanceColumns) {
    seriesData[column] = [];
  }

  for (let i = 0; i < portfolio.index.length; i++) {
    for (const { column, index } of balanceColumns) {
      seriesData[column].push({ time: portfolio.index[i], value: portfolio.data[i][index] });
    }
  }

  const colorPalette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
  ];

  for (let i = 0; i < balanceColumns.length; i++) {
    const { column } = balanceColumns[i];
    const lineSeries = chart.addSeries(LineSeries, { title: column, color: colorPalette[i % colorPalette.length] });
    lineSeries.setData(seriesData[column]);
  }
}

window.setupChart = setupChart;
