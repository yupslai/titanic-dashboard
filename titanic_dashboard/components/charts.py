import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_survival_by_category(df, category, title=None):
    """범주별 생존율을 시각화합니다."""
    survival_rates = df.groupby(category)['Survived'].mean().reset_index()
    survival_rates['Survival_Rate'] = survival_rates['Survived'] * 100
    
    fig = px.bar(
        survival_rates,
        x=category,
        y='Survival_Rate',
        title=title or f'생존율 by {category}',
        labels={'Survival_Rate': '생존율 (%)'},
        text=survival_rates['Survival_Rate'].round(1).astype(str) + '%'
    )
    
    fig.update_traces(textposition='outside')
    return fig

def plot_age_distribution(df):
    """나이 분포를 생존 여부별로 시각화합니다."""
    fig = px.histogram(
        df,
        x='Age',
        color='Survived',
        nbins=20,
        title='나이 분포 by 생존 여부',
        labels={'Survived': '생존'},
        color_discrete_map={0: 'red', 1: 'green'},
        barmode='overlay',
        opacity=0.7
    )
    return fig

def plot_correlation_matrix(df, features):
    """특성 간 상관관계를 히트맵으로 시각화합니다."""
    corr = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=features,
        y=features,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr, 2),
        texttemplate='%{text}',
        textfont={'size': 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='특성 간 상관관계',
        xaxis_title='특성',
        yaxis_title='특성'
    )
    return fig

def plot_feature_importance(importance_df, top_n=10):
    """특성 중요도를 시각화합니다."""
    importance_df = importance_df.head(top_n)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} 중요 특성',
        labels={'importance': '중요도', 'feature': '특성'}
    )
    
    fig.update_traces(texttemplate='%{x:.3f}', textposition='outside')
    return fig

def plot_survival_prediction(df):
    """예측된 생존 확률 분포를 시각화합니다."""
    fig = px.histogram(
        df,
        x='Survival_Probability',
        nbins=30,
        title='생존 확률 분포',
        labels={'Survival_Probability': '생존 확률'}
    )
    
    fig.add_vline(
        x=0.5,
        line_dash='dash',
        line_color='red',
        annotation_text='예측 임계값 (0.5)'
    )
    return fig

def plot_fare_vs_age(df):
    """요금과 나이의 관계를 산점도로 시각화합니다."""
    fig = px.scatter(
        df,
        x='Age',
        y='Fare',
        color='Survived',
        size='Family_Size',
        hover_data=['Name', 'Sex', 'Pclass'],
        title='요금 vs 나이 (생존 여부)',
        labels={'Survived': '생존'},
        color_discrete_map={0: 'red', 1: 'green'}
    )
    return fig

def plot_survival_metrics(metrics):
    """모델 성능 지표를 시각화합니다."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            text=[f'{v:.3f}' for v in metrics.values()],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='모델 성능 지표',
        xaxis_title='지표',
        yaxis_title='점수',
        yaxis_range=[0, 1]
    )
    return fig 