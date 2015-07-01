
	if plots_ and analysistype=="restartUCB":
		id = 109641
		objS = articlesSingle[id]
		objH = articlesHours[id]
		objC = objH
		objM = articlesMultiple[id]

		# tim = np.array(range(shape(articlesSingle[id][0])[0]))
		maxClick = objC.ucbc[-1]
		# calculating batch stats
		with np.errstate(invalid='ignore'):
			accessSingleBatches = np.concatenate((np.array([objC.ucba[0]]), objC.ucba[1:] - objC.ucba[:-1]))
			clickSingleBatches = np.concatenate((np.array([objC.ucbc[0]]), objC.ucbc[1:] - objC.ucbc[:-1]))
			batchChosenCTR = clickSingleBatches / accessSingleBatches

			accessMultBatches = np.concatenate((np.array([objM.ucba[0]]), objM.ucba[1:] - objM.ucba[:-1]))
			clickMultBatches = np.concatenate((np.array([objM.ucbc[0]]), objM.ucbc[1:] - objM.ucbc[:-1]))
			batchMultCTR = clickMultBatches / accessMultBatches

			accessMultrandBatches = np.concatenate((np.array([objM.Oranda[0]]), objM.Oranda[1:] - objM.Oranda[:-1]))
			clickMultrandBatches = np.concatenate((np.array([objM.Orandc[0]]), objM.Orandc[1:] - objM.Orandc[:-1]))
			batchMultrandCTR = clickMultrandBatches / accessMultrandBatches

		yAccMax =max([max(accessMultBatches), max(accessSingleBatches)])
		yCTRMax =max([max(batchMultCTR), max(batchChosenCTR)])
		thetaMax = np.max(objC.theta)
		thetaMin = np.min(objC.theta)
		minTim, maxTim = min(objC.tim), max(objC.tim)
		if 1:
			f, axarr = plt.subplots(4, sharex=True)
			axarr[0].set_ylim([thetaMin, thetaMax])
			axarr[0].plot(objC.tim, objC.theta)

			axarr[0].set_ylabel('Daily \nPreferences')
			axarr[0].set_title("Fluctuating popularity of article")
			# axarr[0].set_title('title ID : ' + str(id) + 'Sessions:' + ','.join([str(x) for x in list(set(objH.resetCount))]) + ' Days:' + ','.join([str(x) for x in list(set(objS.resetCount))]))

			axarr[1].set_ylim([thetaMin, thetaMax])
			axarr[1].plot(objM.tim, objM.theta)
			axarr[1].set_ylabel('Static \nPreferences')

			# axarr[0].set_title('Plots of Daily Dynamic Theta for Min Diff CTR')
			axarr[2].set_ylim([0, yAccMax])
			axarr[2].plot(objC.tim, accessSingleBatches)
			# axarr[2].plot(objC.tim, objS.ucba)
			axarr[2].plot(objM.tim, accessMultBatches, 'm')
			# axarr[2].plot(objM.tim, objM.ucba, 'm')

			axarr[2].set_ylabel('Dynamic \nUCB Access', color='b')
			axarrT = axarr[2].twinx()
			axarrT.plot(objC.tim, batchChosenCTR, 'r')
			axarrT.plot(objM.tim, batchMultCTR, 'g')
			axarrT.plot(objM.tim, batchMultrandCTR, 'y')
			axarrT.set_ylabel('Dynamic CTR', color='r')
			# axarr[3].plot(objC.tim, objC.inPool, '.')
			# axarr[3].plot(objC.tim, objC.selected, '.')
			# axarr[3].plot(objM.tim, objM.inPool, '.')
			# axarr[3].plot(objM.tim, objM.selected, '.')
			# axarr[3].legend(['In Pool', 'Selected'])
			axarr[3].plot(objC.tim, objC.varClus*alpha, objM.tim, objM.varClus*alpha)
			# axarr[4].plot(objS.tim, objM.means)

			axarr[2].set_xlabel('title ID : ' + str(id) + ' Sessions:' + ','.join([str(x) for x in list(set(objH.resetCount))]) + ' Days:' + ','.join([str(x) for x in list(set(objS.resetCount))]))
			# xlabel('days:' + ','.join([str(x) for x in list(set(objC.resetCount))]))
			for t in articlesHours[id].restartTimes:
				if t > minTim and t < maxTim:
					xSet = [t for it in range(31)]
					maxCTR = max([max(batchChosenCTR), max(batchMultCTR), max(batchMultrandCTR)])
					ySet = (np.array(range(0, 31))*1.0/30)*maxCTR
					# axarr[0].plot(xSet, ySet, 'black')
					# axarr[1].plot(xSet, ySet, 'black')
					axarrT.plot(xSet, ySet, 'black')
