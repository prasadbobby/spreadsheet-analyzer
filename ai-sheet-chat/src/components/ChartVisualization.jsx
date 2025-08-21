'use client'

import { useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  LineElement,
  PointElement,
  ArcElement,
} from 'chart.js'
import { Bar, Line, Pie } from 'react-chartjs-2'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { BarChart3, TrendingUp, PieChart } from 'lucide-react'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
)

export default function ChartVisualization({ chartData, title = "Data Visualization" }) {
  if (!chartData) return null

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: title,
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgba(133, 0, 223, 1)',
        borderWidth: 1,
      }
    },
    scales: chartData.type !== 'pie' ? {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)',
        }
      },
      x: {
        grid: {
          display: false
        }
      }
    } : undefined
  }

  const getChartIcon = (type) => {
    switch (type) {
      case 'line': return TrendingUp
      case 'pie': return PieChart
      default: return BarChart3
    }
  }

  const ChartIcon = getChartIcon(chartData.type)

  const renderChart = () => {
    switch (chartData.type) {
      case 'line':
        return <Line data={chartData.data} options={chartOptions} />
      case 'pie':
        return <Pie data={chartData.data} options={chartOptions} />
      default:
        return <Bar data={chartData.data} options={chartOptions} />
    }
  }

  return (
    <Card className="border-0 shadow-lg bg-white">
      <CardHeader className="pb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
            <ChartIcon className="h-4 w-4 text-purple-600" />
          </div>
          <CardTitle className="text-lg font-semibold text-gray-900">
            {title}
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-80 w-full">
          {renderChart()}
        </div>
      </CardContent>
    </Card>
  )
}