# MMA3001 Project Report

## 1. Engineering Problem

### 1.1 Engineering Context and Motivation

Currently, most industrial Heating, Ventilation and Air-Conditioning (HVAC) systems operate on a Demand Controlled Ventilattion (DCV) system, where a snesor will notice someone enter the room, track changes in C02 levels and apply appropriate cooling and heating. However, while the system works, it would be more comfortable and more efficient if the HVAC system begun operating at appropriate levels prior to anyone even walking into the room. 

If the HVAC system assessed room layout, time of the day, weather, temperature and future headcounts, the system could potentially begin cooling, heating or ventalating a room before anyone enters. This not only creates a more comfortable environment, but if the system knew where and when to ventilate, then power can be reduces when its not needed, creating an overall more efficient system.

However, the project itself only focuses on one part of developing such a system, using past room occupancy to forecast 30 minutes into the future.

### 1.2 Problem Definition

This project investigates whether historical building occupancy measurements can predict the number of occupants in a monitored space 30 minutes into the future. The project will compare a simple forecasting baseline against a TBD regresion method to determine if the model offers any predictive accuracy and assess computational performance

**Research question:**

How accurately can historical occupancy measurements forecast building space headcount 30 minutes ahead, and how do different regression approaches compare?

### 1.3 Intended Application

The forecast is intended to provide information that could support future occupancy aware ventilation demand planning. However, the project is not intended to simulate the HVAC system, calculate energy savings or directly control ventilation equipment.

### 1.4 Scope and Limitations

- The initial investigation focuses on one monitored building space, with possible evaluation across additional spaces.
- The project predicts sensor reported occupancy rather than independently verified physical headcounts.
- Data limitations include irregular recording intervals, extended periods without observations and uncertain physical room mappings.

## 2. Inputs and Outputs

### 2.1 Input Dataset

The project uses Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data.

The occupancy dataset contains approximately 9.09 million records across five building-space identifiers.

Relevant variables include:

| Variable | Meaning | Format or units |
|---|---|---|
| `floorspaceid` | Identifies a monitored building space | Text identifier |
| `collecteddate` | Recorded observation timestamp | Date and time with UTC offset |
| `headcount` | Sensor-reported number of occupants | People |
| `occupancystatus` | Reported occupancy state | Categorical |

### 2.2 Model Inputs

- The proposed model will use historical headcounts and potentially time related features such as time of day and day of week.
- The final number of previous observations used, selected building space and training period remain to be determined.

### 2.3 Model Outputs

- The primary output will be a predicted headcount for the selected building space 30 minutes after the forecast starting time.
- The output represents a predicted number of people based on the available sensor data.
- Model accuracy will be assessed by comparing predictions against subsequently recorded headcounts using suitable regression evaluation metrics.

### 2.4 Data Preparation and Invalid Inputs

An initial feasibility test was performed for October 2024, which assessed the availability of usable 30 minute forecasting examples. For the analysis, observations were accepted only when a recorded headcount was available within five minutes before the required input or target time. However the five minute threshold is subject to change and any missing or unsuported data wont be included in the model.