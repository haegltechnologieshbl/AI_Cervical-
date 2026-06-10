/**
 * Dashboard Charts Module
 * Provides Chart.js integration for hospital management dashboard
 */

class DashboardCharts {
    constructor() {
        this.charts = {};
        this.colors = {
            normal: '#22c55e',
            ascus: '#84cc16',
            lsil: '#eab308',
            hsil: '#f97316',
            carcinoma: '#ef4444'
        };
        this.stageLabels = ['Normal', 'ASC-US', 'LSIL', 'HSIL', 'Carcinoma'];
    }

    /**
     * Initialize Stage Distribution Bar Chart
     */
    initStageDistribution(canvasId, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.warn(`Canvas with id "${canvasId}" not found`);
            return;
        }

        if (this.charts.stage) {
            this.charts.stage.destroy();
        }

        this.charts.stage = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.stageLabels,
                datasets: [{
                    label: 'Number of Cases',
                    data: [
                        data[0] || 0,
                        data[1] || 0,
                        data[2] || 0,
                        data[3] || 0,
                        data[4] || 0
                    ],
                    backgroundColor: [
                        this.colors.normal,
                        this.colors.ascus,
                        this.colors.lsil,
                        this.colors.hsil,
                        this.colors.carcinoma
                    ],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw} cases`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { precision: 0 },
                        grid: { color: '#f3f4f6' }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });
    }

    /**
     * Initialize Monthly Detection Rate Line Chart
     */
    initDetectionRate(canvasId, labels, data) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.warn(`Canvas with id "${canvasId}" not found`);
            return;
        }

        if (this.charts.detection) {
            this.charts.detection.destroy();
        }

        this.charts.detection = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Detection Rate (%)',
                    data: data,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#3b82f6',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Rate: ${context.raw.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        grid: { color: '#f3f4f6' }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });
    }

    /**
     * Initialize Follow-Up Status Doughnut Chart
     */
    initFollowUpStatus(canvasId, pending, overdue, completed) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) {
            console.warn(`Canvas with id "${canvasId}" not found`);
            return;
        }

        if (this.charts.followup) {
            this.charts.followup.destroy();
        }

        this.charts.followup = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Pending', 'Overdue', 'Completed'],
                datasets: [{
                    data: [pending, overdue, completed],
                    backgroundColor: ['#eab308', '#ef4444', '#22c55e'],
                    borderWidth: 0,
                    spacing: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 15
                        }
                    }
                },
                cutout: '70%'
            }
        });
    }

    /**
     * Destroy all charts
     */
    destroyAll() {
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        this.charts = {};
    }
}

/**
 * Initialize dashboard on page load
 * NOTE: This is disabled for admin dashboard since admin_dashboard.html
 * has its own chart initialization logic. This is used for user dashboard.
 */
document.addEventListener('DOMContentLoaded', async function() {
    // Skip auto-initialization on admin dashboard pages
    // The admin dashboard handles its own chart initialization to avoid conflicts
    if (window.location.pathname.startsWith('/admin/')) {
        console.log('Admin dashboard detected, skipping auto-initialization');
        return;
    }

    const dashboardCharts = new DashboardCharts();

    try {
        // Fetch hospital dashboard statistics
        const response = await fetch('/api/v1/dashboard/hospital-stats');

        if (!response.ok) {
            throw new Error('Failed to fetch dashboard statistics');
        }

        const stats = await response.json();

        // Update statistics cards
        updateDashboardStats(stats);

        // Initialize stage distribution chart
        if (document.getElementById('stageDistributionChart')) {
            dashboardCharts.initStageDistribution('stageDistributionChart', stats.stage_distribution);
        }

        // Initialize follow-up status chart
        if (document.getElementById('followUpChart')) {
            const followUpData = stats.follow_up_stats || {};
            dashboardCharts.initFollowUpStatus(
                'followUpChart',
                followUpData.pending || 0,
                followUpData.overdue || 0,
                (followUpData.pending || 0) - (followUpData.overdue || 0)
            );
        }

        // Initialize monthly detection rate chart (if trend data is available)
        if (document.getElementById('detectionRateChart')) {
            // For now, show current month's rate
            const monthlyRate = stats.monthly_detection_rate || 0;
            const currentMonth = new Date().toLocaleDateString('en-US', { month: 'short' });
            dashboardCharts.initDetectionRate('detectionRateChart', [currentMonth], [monthlyRate]);
        }

        console.log('Dashboard initialized successfully');

    } catch (error) {
        console.error('Failed to initialize dashboard:', error);

        // Show error message on page
        const errorAlert = document.getElementById('dashboard-error-alert');
        if (errorAlert) {
            errorAlert.classList.remove('hidden');
            errorAlert.querySelector('.error-message').textContent = 'Failed to load dashboard statistics';
        }
    }
});

/**
 * Update dashboard statistics cards with fetched data
 */
function updateDashboardStats(stats) {
    // Patient Statistics
    updateStatCard('total-patients', stats.patient_stats?.total_registered);
    updateStatCard('total-screened', stats.patient_stats?.total_screened);
    updateStatCard('high-risk-patients', stats.patient_stats?.high_risk);
    updateStatCard('recent-visits', stats.patient_stats?.recent_visits);

    // Case Statistics
    updateStatCard('positive-cases', stats.case_stats?.positive_cases);
    updateStatCard('negative-cases', stats.case_stats?.negative_cases);
    updateStatCard('pending-reviews', stats.case_stats?.pending_reviews);

    // Appointment Statistics
    updateStatCard('todays-appointments', stats.appointment_stats?.today);
    updateStatCard('weekly-appointments', stats.appointment_stats?.this_week);

    // Follow-Up Statistics
    updateStatCard('pending-followups', stats.follow_up_stats?.pending);
    updateStatCard('overdue-followups', stats.follow_up_stats?.overdue);

    // Monthly Detection Rate
    updateStatCard('detection-rate', stats.monthly_detection_rate, '%');
}

/**
 * Update individual statistics card
 */
function updateStatCard(elementId, value, suffix = '') {
    const element = document.getElementById(elementId);
    if (element && value !== null && value !== undefined) {
        element.textContent = value + suffix;
    }
}
